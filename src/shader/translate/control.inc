// keylime: the program body: blocks, branches and loops (part of the RDNA2 to SPIR-V translator, inside its class; included by translate.cpp).
    // ------------------------------------------------------------ body ----
    // Control flow: `pc` holds the next block's number. Blocks are emitted in program order,
    // each run when pc names it, so forward branches need no loop. Backward branches make
    // loop regions (overlapping ones merged), each a structured loop that runs again while pc
    // stays inside it. Ending the program sets pc past the last block.
    struct Region {
        uint32_t lo, hi;
        bool active = false;
    };
    std::vector<Region> regions;
    std::vector<size_t> first_inst;
    Id trips = 0;
    uint32_t end_block = 0;

    uint32_t trip_limit() const { return p.stage == HwStage::Ps ? 512 : p.stage == HwStage::Vs || p.stage == HwStage::Ms ? 1024 : 8192; }

    void find_regions() {
        std::vector<std::pair<uint32_t, uint32_t>> r;
        for (auto& i : insts) {
            uint32_t target;
            if (!gfx10::branch_target(i, target)) continue;
            auto tb = block_of.find(target);
            if (tb == block_of.end()) continue;
            uint32_t src = std::prev(block_of.upper_bound(i.offset))->second;
            if (tb->second <= src) r.push_back({tb->second, src});
        }
        for (bool changed = true; changed;) {
            changed = false;
            for (size_t x = 0; x < r.size() && !changed; ++x)
                for (size_t y = 0; y < r.size() && !changed; ++y) {
                    if (x == y) continue;
                    auto& a = r[x];
                    auto& z = r[y];
                    bool same = a == z;
                    bool crossing = a.first < z.first && z.first <= a.second && a.second < z.second;
                    if (!same && !crossing) continue;
                    a = {std::min(a.first, z.first), std::max(a.second, z.second)};
                    r.erase(r.begin() + y);
                    changed = true;
                }
        }
        for (auto& [lo, hi] : r) regions.push_back({lo, hi});
    }

    Id pc_in(uint32_t lo, uint32_t hi) {
        return b.op(spv::OpULessThanEqual, t_bool, {isub(load(pc, t_u32), u(lo)), u(hi - lo)});
    }

    // Lane-bit facts into a block: those every emitted predecessor leaves. A predecessor emitted
    // later (a back edge) is assumed to keep them, and checked once all is emitted: a block whose
    // assumption failed is translated again without it (`distrust`).
    std::vector<std::vector<uint32_t>> preds;
    std::vector<std::bitset<SGPRS>> exit_mine, assumed;
    std::vector<char> emitted;
    std::set<uint32_t> distrust, broken;
    void block(uint32_t bi, uint32_t start) {
        if (bi || !preds[bi].empty()) {
            std::bitset<SGPRS> in;
            in.set();
            bool any = false, later = false;
            for (uint32_t q : preds[bi]) {
                if (emitted[q]) in &= exit_mine[q], any = true;
                else later = true;
            }
            if (bi == 0) in &= mine_reg, any = true;
            if (!any || (later && distrust.count(bi))) in.reset();
            assumed[bi] = in;
            mine_reg = in;
        }
        block_body(bi, start);
        exit_mine[bi] = mine_reg, emitted[bi] = 1;
    }
    void check_assumptions() {
        for (uint32_t bi = 0; bi < end_block; ++bi)
            for (uint32_t q : preds[bi])
                if (emitted[q] && emitted[bi] && (exit_mine[q] & assumed[bi]) != assumed[bi]) broken.insert(bi);
    }
    void block_body(uint32_t bi, uint32_t start) {
        size_t ii = first_inst[bi];
        for (; ii < insts.size(); ++ii) {
            const Inst& i = insts[ii];
            auto next_block = block_of.find(i.offset);
            if (i.offset != start && next_block != block_of.end()) {
                store(pc, u(next_block->second));
                return;
            }
            uint32_t target;
            if (gfx10::ends_program(i)) {
                store(pc, u(end_block));
                return;
            }
            if (gfx10::branch_target(i, target)) {
                auto tb = block_of.find(target);
                auto fb = block_of.find(i.offset + i.size);
                uint32_t tn = tb != block_of.end() ? tb->second : end_block, fn_ = fb != block_of.end() ? fb->second : end_block;
                std::string_view n = i.name;
                if (n == "s_branch") return store(pc, u(tn));
                Id c = 0;
                if (n == "s_cbranch_scc0") c = b.op(spv::OpLogicalNot, t_bool, {get_scc()});
                else if (n == "s_cbranch_scc1") c = get_scc();
                else if (n == "s_cbranch_vccz") c = b.op(spv::OpIEqual, t_bool, {mask64(VCC_LO), u(0)});
                else if (n == "s_cbranch_vccnz") c = ne0(mask64(VCC_LO));
                else if (n == "s_cbranch_execz") c = b.op(spv::OpIEqual, t_bool, {mask64(EXEC_LO), u(0)});
                else if (n == "s_cbranch_execnz") c = ne0(mask64(EXEC_LO));
                else {
                    unsupported(i);
                    c = b.c_bool(false);
                }
                return store(pc, sel(t_u32, c, u(tn), u(fn_)));
            }
            emit(i);
        }
        store(pc, u(end_block));
    }

    void if_then(Id cond, const std::function<void()>& f) {
        Id yes = b.id(), after = b.id();
        b.op_void(spv::OpSelectionMerge, {after, 0});
        b.op_void(spv::OpBranchConditional, {cond, yes, after});
        b.label(yes);
        f();
        b.op_void(spv::OpBranch, {after});
        b.label(after);
    }

    void loop(Region& r, const std::vector<uint32_t>& starts) {
        loop(r, [&] { range(r.lo, r.hi, starts, false); });
    }

    void loop(Region& r, const std::function<void()>& inner) {
        r.active = true;
        if_then(pc_in(r.lo, r.hi), [&] {
            Id header = b.id(), body_ = b.id(), cont = b.id(), merge = b.id();
            // Each loop counts its own trips: with one shared counter an inner loop reset the
            // outer one's on every pass, so an outer loop around an inner one had no cap at all.
            Id trips = b.local(p_fn_u32, u(0));
            store(trips, u(0));
            b.op_void(spv::OpBranch, {header});
            b.label(header);
            b.op_void(spv::OpLoopMerge, {merge, cont, 0});
            b.op_void(spv::OpBranch, {body_});
            b.label(body_);
            inner();
            b.op_void(spv::OpBranch, {cont});
            b.label(cont);
            Id t = iadd(load(trips, t_u32), u(1));
            store(trips, t);
            Id again = b.op(spv::OpLogicalAnd, t_bool, {pc_in(r.lo, r.hi), b.op(spv::OpULessThan, t_bool, {t, u(trip_limit())})});
            b.op_void(spv::OpBranchConditional, {again, header, merge});
            b.label(merge);
            // A loop stopped by its cap ends the program (as VIctory's forks do): carrying on let an
            // outer loop enter it again with a fresh count, so nested caps multiplied (8192 x 8192
            // trips) and a loop spinning on bad data held the GPU until Windows reset it.
            Id capped = b.op(spv::OpUGreaterThanEqual, t_bool, {load(trips, t_u32), u(trip_limit())});
            store(pc, sel(t_u32, capped, u(end_block), load(pc, t_u32)));
        });
        r.active = false;
    }

    std::vector<cfg::Term> terms() const {
        std::vector<cfg::Term> out(end_block);
        for (uint32_t bi = 0; bi < end_block; ++bi) {
            cfg::Term& t = out[bi];
            t.kind = cfg::Exit::End;
            for (size_t ii = first_inst[bi]; ii < insts.size(); ++ii) {
                const Inst& i = insts[ii];
                if (ii != first_inst[bi] && block_of.count(i.offset)) {
                    t.kind = cfg::Exit::Next;
                    break;
                }
                if (gfx10::ends_program(i)) break;
                uint32_t target;
                if (!gfx10::branch_target(i, target)) continue;
                auto tb = block_of.find(target);
                auto fb = block_of.find(i.offset + i.size);
                t.t = tb != block_of.end() ? tb->second : end_block;
                t.f = fb != block_of.end() ? fb->second : end_block;
                t.kind = std::string_view(i.name) == "s_branch" ? cfg::Exit::Jump : cfg::Exit::Branch;
                break;
            }
        }
        return out;
    }

    void if_else(Id cond, const std::function<void()>& f, const std::function<void()>& g) {
        Id yes = b.id(), no = b.id(), after = b.id();
        b.op_void(spv::OpSelectionMerge, {after, 0});
        b.op_void(spv::OpBranchConditional, {cond, yes, no});
        b.label(yes);
        f();
        b.op_void(spv::OpBranch, {after});
        b.label(no);
        g();
        b.op_void(spv::OpBranch, {after});
        b.label(after);
    }

    void count_plan(const std::vector<cfg::Node>& nodes) {
        out.blocks = end_block;
        std::function<void(const std::vector<cfg::Node>&)> walk = [&](const std::vector<cfg::Node>& ns) {
            for (auto& x : ns) {
                if (x.kind == cfg::Node::Range) out.guarded_blocks += x.b - x.a + 1;
                walk(x.body);
                walk(x.other);
            }
        };
        walk(nodes);
    }

    Id pc_is(uint32_t k) { return b.op(spv::OpIEqual, t_bool, {load(pc, t_u32), u(k)}); }
    // A guard in front of block a; b > a when loops start at a (pc may name any block in them).
    Id pc_at(const cfg::Node& x) { return x.b > x.a ? pc_in(x.a, x.b) : pc_is(x.a); }

    void run(const std::vector<cfg::Node>& nodes, const std::vector<uint32_t>& starts) {
        for (auto& x : nodes) {
            switch (x.kind) {
                case cfg::Node::Block: block(x.a, starts[x.a]); break;
                case cfg::Node::If: if_then(pc_at(x), [&] { run(x.body, starts); }); break;
                case cfg::Node::IfElse: if_else(pc_at(x), [&] { run(x.body, starts); }, [&] { run(x.other, starts); }); break;
                case cfg::Node::Range: range(x.a, x.b, starts, false); break;
                case cfg::Node::Loop: loop(regions[x.a], [&] { run(x.body, starts); }); break;
            }
        }
    }

    void range(uint32_t lo, uint32_t hi, const std::vector<uint32_t>& starts, bool top) {
        for (uint32_t k = lo; k <= hi;) {
            Region* outer = nullptr;
            for (auto& r : regions)
                if (!r.active && r.lo == k && r.hi <= hi && (!outer || r.hi > outer->hi)) outer = &r;
            if (outer) {
                uint32_t next = outer->hi + 1;
                loop(*outer, starts);
                k = next;
                continue;
            }
            if (top && k == 0) block(0, starts[0]);
            else if_then(b.op(spv::OpIEqual, t_bool, {load(pc, t_u32), u(k)}), [&] { block(k, starts[k]); });
            ++k;
        }
    }

    bool body() {
        Id fn = b.id();
        b.begin_function(fn, t_void, t_fn);
        b.label();
        scc = b.local(p_fn_bool, b.c_bool(false));
        registers.insert(scc);
        pc = b.local(p_fn_u32, u(0));
        trips = b.local(p_fn_u32, u(0));
        prologue();
        std::vector<uint32_t> starts;
        for (auto& [off, n] : block_of) {
            starts.push_back(off);
            size_t ii = 0;
            while (ii < insts.size() && insts[ii].offset < off) ++ii;
            first_inst.push_back(ii);
        }
        end_block = uint32_t(starts.size());
        find_regions();
        std::vector<cfg::Region> cr;
        for (auto& r : regions) cr.push_back({r.lo, r.hi});
        auto tm = terms();
        preds.assign(end_block, {}), exit_mine.assign(end_block, {}), assumed.assign(end_block, {}), emitted.assign(end_block, 0);
        for (uint32_t bi = 0; bi < end_block; ++bi) {
            auto to = [&](uint32_t t) { if (t < end_block) preds[t].push_back(bi); };
            if (tm[bi].kind == cfg::Exit::Next) to(bi + 1);
            if (tm[bi].kind == cfg::Exit::Jump) to(tm[bi].t);
            if (tm[bi].kind == cfg::Exit::Branch) to(tm[bi].t), to(tm[bi].f);
        }
        auto nodes = cfg::plan(tm, cr, &out.plan_note);
        count_plan(nodes);
        run(nodes, starts);
        if (p.stage == HwStage::Ms) mesh_epilogue();
        b.op_void(spv::OpReturn, {});
        b.end_function();
        // Entry point.
        spv::ExecutionModel model = p.stage == HwStage::Vs ? spv::ExecutionModelVertex
                                  : p.stage == HwStage::Ps ? spv::ExecutionModelFragment
                                  : p.stage == HwStage::Ms ? spv::ExecutionModelMeshEXT
                                                           : spv::ExecutionModelGLCompute;
        b.entry(model, fn, "main", interface);
        if (p.stage == HwStage::Ps) b.mode(fn, spv::ExecutionModeOriginUpperLeft);
        if (p.stage == HwStage::Ms) {
            out.local_size[0] = wide ? (p.ms_threads + 63) / 64 * 32 : p.ms_threads;  // wide: two console threads per invocation
            b.mode(fn, spv::ExecutionModeLocalSize, {out.local_size[0], 1, 1});
            b.mode(fn, spv::ExecutionModeOutputVertices, {ms_vertices()});
            b.mode(fn, spv::ExecutionModeOutputPrimitivesEXT, {ms_primitives()});
            b.mode(fn, ms_lines() ? spv::ExecutionModeOutputLinesEXT : spv::ExecutionModeOutputTrianglesEXT);
        }
        if (p.stage == HwStage::Cs) {
            for (int k = 0; k < 3; ++k) out.local_size[k] = p.cs_threads[k] ? p.cs_threads[k] : 1;
            if (wide) {  // half as many invocations, each running two lanes, in whole 32-lane waves
                uint32_t threads = out.local_size[0] * out.local_size[1] * out.local_size[2];
                out.local_size[0] = (threads + 63) / 64 * 32, out.local_size[1] = out.local_size[2] = 1;
            }
            b.mode(fn, spv::ExecutionModeLocalSize, {out.local_size[0], out.local_size[1], out.local_size[2]});
        }
        return true;
    }


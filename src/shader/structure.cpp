#include "shader/structure.h"

#include <algorithm>
#include <cstdio>

namespace kl::shader::cfg {
namespace {

constexpr uint32_t ANY = ~0u;

// Every block runs under a check of pc (the next block's number), so a loop entered in its
// middle still nests: an edge from outside a loop to a block inside it counts, for planning,
// as an edge to the loop's first block (the loop runs while pc is anywhere in it), and the
// loop's body is split into parts at the blocks entered from outside, each run when pc names
// it. Guards in front of a loop check pc against the loop's whole range for the same reason.
struct Planner {
    std::vector<Term> terms;  // with edges into a loop's middle from outside it moved to the loop's first block
    const std::vector<Region>& regions;
    uint32_t n;
    std::vector<std::vector<uint32_t>> preds;
    std::vector<std::vector<uint32_t>> entries;  // per loop: blocks after its first entered from outside it
    std::vector<char> active;
    std::string why;

    static bool inside(const Region& r, uint32_t k) { return r.lo <= k && k <= r.hi; }

    // Where an edge from `from` to `to` lands for planning: the first block of the largest loop
    // that holds `to` past its first block and does not hold `from`.
    uint32_t land(uint32_t from, uint32_t to) const {
        int best = -1;
        for (size_t r = 0; r < regions.size(); ++r) {
            auto& g = regions[r];
            if (g.lo < to && to <= g.hi && !inside(g, from) && (best < 0 || g.hi - g.lo > regions[size_t(best)].hi - regions[size_t(best)].lo))
                best = int(r);
        }
        return best < 0 ? to : regions[size_t(best)].lo;
    }

    Planner(const std::vector<Term>& t, const std::vector<Region>& r) : terms(t), regions(r), n(uint32_t(t.size())) {
        entries.resize(r.size());
        for (uint32_t k = 0; k < n; ++k) {
            Term& x = terms[k];
            const auto ss = real_succ(t[k]);
            for (size_t ri = 0; ri < r.size() && !ss.empty(); ++ri) {
                auto& g = r[ri];
                if (inside(g, k)) continue;
                for (uint32_t s : ss) {
                    if (!(g.lo < s && s <= g.hi)) continue;
                    // As seen from inside this loop: the first block of the largest loop within it holding s past its first block.
                    uint32_t e = s;
                    uint32_t size = 0;
                    for (auto& c : r)
                        if (&c != &g && c.lo >= g.lo && c.hi <= g.hi && c.lo < s && s <= c.hi && c.hi - c.lo >= size && !(c.lo == g.lo && c.hi == g.hi))
                            e = c.lo, size = c.hi - c.lo;
                    if (e != g.lo && !has(entries[ri], e)) entries[ri].push_back(e);
                }
            }
            if (x.kind == Exit::Jump || x.kind == Exit::Branch) {
                x.t = land(k, std::min(x.t, n));
                if (x.kind == Exit::Branch) x.f = land(k, std::min(x.f, n));
            }
        }
        for (auto& e : entries) std::sort(e.begin(), e.end());
        preds.resize(n + 1);
        active.resize(r.size());
        for (uint32_t k = 0; k < n; ++k)
            for (uint32_t s : succ(k)) preds[std::min(s, n)].push_back(k);
    }

    std::vector<uint32_t> real_succ(const Term& t) const {
        switch (t.kind) {
            case Exit::Jump: return {std::min(t.t, n)};
            case Exit::Branch: return {std::min(t.t, n), std::min(t.f, n)};
            default: return {};
        }
    }

    // The pc values that reach block k through a guard in front of it: k, or, if loops not yet
    // being planned start at k, anywhere up to the end of the largest of them.
    uint32_t span(uint32_t k) const {
        uint32_t hi = k;
        for (size_t r = 0; r < regions.size(); ++r)
            if (!active[r] && regions[r].lo == k) hi = std::max(hi, regions[r].hi);
        return hi;
    }
    Node guard(Node::Kind kind, uint32_t k) const {
        Node g{kind, k};
        g.b = span(k);
        return g;
    }

    std::vector<uint32_t> succ(uint32_t k) const {
        const Term& t = terms[k];
        switch (t.kind) {
            case Exit::Next: return {k + 1};
            case Exit::End: return {n};
            case Exit::Jump: return {std::min(t.t, n)};
            case Exit::Branch: return {std::min(t.t, n), std::min(t.f, n)};
        }
        return {n};
    }

    static bool has(const std::vector<uint32_t>& v, uint32_t x) { return std::find(v.begin(), v.end(), x) != v.end(); }

    bool nested_loop_holds(uint32_t lo, uint32_t hi, uint32_t from, uint32_t to) const {
        for (size_t r = 0; r < regions.size(); ++r)
            if (!active[r] && regions[r].lo >= lo && regions[r].hi <= hi && regions[r].lo <= to && from <= regions[r].hi) return true;
        return false;
    }

    template <class... A>
    void note(const char* f, A... a) {
        if (!why.empty()) return;
        char b[160];
        snprintf(b, sizeof b, f, a...);
        why = b;
    }

    bool valid(uint32_t lo, uint32_t hi, uint32_t exit, uint32_t entry, const std::vector<uint32_t>& esc) const {
        if (lo > hi) return true;
        for (size_t r = 0; r < regions.size(); ++r) {
            if (active[r]) continue;
            auto& g = regions[r];
            bool overlaps = g.lo <= hi && g.hi >= lo;
            if (overlaps && (g.lo < lo || g.hi > hi)) return false;
        }
        for (uint32_t k = lo; k <= hi; ++k) {
            for (uint32_t s : succ(k)) {
                if (s <= k) {
                    if (has(esc, s) || nested_loop_holds(lo, hi, k, s)) continue;
                    return false;
                }
                if (s <= hi || s == exit || s == n || has(esc, s)) continue;
                return false;
            }
            for (uint32_t p : preds[k]) {
                if (p >= lo && p <= hi) continue;
                if (k == lo && (entry == ANY || p == entry)) continue;
                return false;
            }
        }
        return true;
    }

    int region_at(uint32_t k, uint32_t hi) const {
        int best = -1;
        for (size_t r = 0; r < regions.size(); ++r)
            if (!active[r] && regions[r].lo == k && regions[r].hi <= hi && (best < 0 || regions[r].hi > regions[size_t(best)].hi))
                best = int(r);
        return best;
    }

    bool rest(uint32_t k, uint32_t hi, uint32_t exit, const std::vector<uint32_t>& esc, std::vector<Node>& out) {
        if (k > hi) return true;
        Node g = guard(Node::If, k);
        seq(k, hi, exit, esc, g.body);
        out.push_back(std::move(g));
        return true;
    }

    // Blocks lo..hi that do not nest as one sequence, as few parts as possible, each run when pc
    // names its first block (as Range does block by block). A part may leave to any later or
    // earlier part's first block. `cuts`: blocks that must start a part.
    bool split(uint32_t lo, uint32_t hi, uint32_t exit, const std::vector<uint32_t>& esc, std::vector<Node>& out, std::vector<uint32_t> cuts = {}) {
        if (lo > hi) return true;
        // A part never starts inside a loop that is not being planned: it starts at the loop instead.
        auto at = [&](uint32_t x) {
            uint32_t best = x, size = 0;
            for (size_t r = 0; r < regions.size(); ++r) {
                auto& g = regions[r];
                if (!active[r] && g.lo < x && x <= g.hi && g.lo >= lo && g.hi - g.lo >= size) best = g.lo, size = g.hi - g.lo;
            }
            return best;
        };
        std::vector<char> cut(hi - lo + 1, 0);
        cut[0] = 1;
        for (uint32_t c : cuts)
            if (c >= lo && c <= hi) cut[at(c) - lo] = 1;
        for (bool changed = true; changed;) {
            changed = false;
            auto add = [&](uint32_t x) {
                x = at(x);
                if (x >= lo && x <= hi && !cut[x - lo]) cut[x - lo] = 1, changed = true;
            };
            for (size_t r = 0; r < regions.size(); ++r) {
                auto& g = regions[r];
                if (active[r] || g.hi < lo || g.lo > hi) continue;
                if (g.lo < lo || g.hi > hi) {
                    if (!(g.lo < lo && g.hi >= hi)) return false;  // a loop crossing the range's ends
                    continue;
                }
                if (g.hi + 1 <= hi) add(g.hi + 1);
                add(g.lo);
            }
            uint32_t a = lo;
            for (uint32_t k = lo; k <= hi; ++k) {
                if (cut[k - lo]) a = k;
                for (uint32_t s : succ(k)) {
                    if (s >= lo && s <= hi) {
                        if (s > k) {
                            // Forward inside the range: fine within this part only if nothing in
                            // between starts a new part.
                            bool crosses = false;
                            for (uint32_t x = k + 1; x <= s && !crosses; ++x) crosses = cut[x - lo] && x != s;
                            if (crosses) add(s);
                        } else if (!has(esc, s) && !nested_loop_holds(lo, hi, k, s) && !cut[s - lo]) {
                            add(s);
                        }
                    } else if (s != exit && s != n && !has(esc, s)) {
                        return false;
                    }
                }
                if (k != a)
                    for (uint32_t p : preds[k])
                        if (p < a || p > k) {
                            // Entered from outside its part: it must start one (unless a loop inside holds both).
                            if (!(p > k && nested_loop_holds(a, hi, p, k))) add(k);
                            break;
                        }
            }
        }
        std::vector<uint32_t> starts;
        for (uint32_t k = lo; k <= hi; ++k)
            if (cut[k - lo]) starts.push_back(k);
        std::vector<uint32_t> part_esc = esc;
        part_esc.insert(part_esc.end(), starts.begin(), starts.end());
        part_esc.push_back(exit);
        for (size_t i = 0; i < starts.size(); ++i) {
            uint32_t a = starts[i], z = i + 1 < starts.size() ? starts[i + 1] - 1 : hi;
            if (!valid(a, z, i + 1 < starts.size() ? z + 1 : exit, ANY, part_esc)) return false;
        }
        for (size_t i = 0; i < starts.size(); ++i) {
            uint32_t a = starts[i], z = i + 1 < starts.size() ? starts[i + 1] - 1 : hi;
            Node part = guard(Node::If, a);
            seq(a, z, i + 1 < starts.size() ? z + 1 : exit, part_esc, part.body);
            out.push_back(std::move(part));
        }
        return true;
    }

    bool fallback(uint32_t k, uint32_t hi, uint32_t exit, const std::vector<uint32_t>& esc, std::vector<Node>& out) {
        if (k > hi) return true;
        std::vector<Node> parts;
        if (split(k, hi, exit, esc, parts)) {
            for (auto& p : parts) out.push_back(std::move(p));
            return true;
        }
        note("block %u: %s", k ? k - 1 : 0, terms[k ? k - 1 : 0].kind == Exit::Jump ? "a jump" : "a branch");
        out.push_back({Node::Range, k, hi});
        return true;
    }
    bool exits_ok(const Region& r, const std::vector<uint32_t>& out_ok) const {
        for (uint32_t x = r.lo; x <= r.hi; ++x)
            for (uint32_t s : succ(x))
                if ((s < r.lo || s > r.hi) && s != r.hi + 1 && s != n && !has(out_ok, s)) return false;
        return true;
    }

    bool seq(uint32_t lo, uint32_t hi, uint32_t exit, const std::vector<uint32_t>& esc, std::vector<Node>& out) {
        for (uint32_t k = lo; k <= hi;) {
            if (int ri = region_at(k, hi); ri >= 0) {
                const Region& r = regions[size_t(ri)];
                std::vector<uint32_t> out_ok = esc;
                out_ok.push_back(exit);
                // Exits past the block after the loop: fine if they land later in this sequence,
                // which is then split into parts at them.
                std::vector<uint32_t> far;
                bool far_ok = true;
                for (uint32_t x = r.lo; x <= r.hi; ++x)
                    for (uint32_t s : succ(x))
                        if ((s < r.lo || s > r.hi) && s != r.hi + 1 && s != n && !has(out_ok, s) && !has(far, s)) {
                            far.push_back(s);
                            far_ok &= s > r.hi + 1 && s <= hi;
                        }
                if (!far_ok) {
                    note("the loop at blocks %u-%u leaves to a block other than the one after it", r.lo, r.hi);
                    return fallback(k, hi, exit, esc, out);
                }
                out_ok.insert(out_ok.end(), far.begin(), far.end());
                Node l{Node::Loop, uint32_t(ri)};
                active[size_t(ri)] = 1;
                std::vector<uint32_t> inner = out_ok;
                inner.push_back(r.lo);
                inner.push_back(r.hi + 1);
                // Entered in the middle, or not nesting as one sequence: parts, each run when pc
                // names its first block (one per block entered from outside, at least).
                const auto& ent = entries[size_t(ri)];
                if (ent.empty() && valid(r.lo, r.hi, r.hi + 1, ANY, inner)) seq(r.lo, r.hi, r.hi + 1, inner, l.body);
                else if (!split(r.lo, r.hi, r.hi + 1, inner, l.body, ent)) {
                    note("the loop at blocks %u-%u does not nest", r.lo, r.hi);
                    l.body.clear();
                    l.body.push_back({Node::Range, r.lo, r.hi});
                }
                active[size_t(ri)] = 0;
                out.push_back(std::move(l));
                if (!far.empty()) {
                    std::vector<Node> parts;
                    if (split(r.hi + 1, hi, exit, esc, parts, far)) for (auto& p : parts) out.push_back(std::move(p));
                    else {
                        note("the blocks after the loop at blocks %u-%u do not nest", r.lo, r.hi);
                        out.push_back({Node::Range, r.hi + 1, hi});
                    }
                    return true;
                }
                return rest(r.hi + 1, hi, exit, esc, out);
            }
            out.push_back({Node::Block, k});
            const Term& t = terms[k];
            if (t.kind == Exit::End) return true;
            if (t.kind == Exit::Next) {
                ++k;
                continue;
            }
            if (t.kind == Exit::Jump) {
                if (t.t > k && t.t <= hi) {
                    k = t.t;
                    continue;
                }
                return t.t != exit;
            }
            if (t.f != k + 1) return fallback(k + 1, hi, exit, esc, out);
            uint32_t tt = std::min(t.t, n);
            if (tt == k + 1) {
                ++k;
                continue;
            }
            if (tt <= k || tt > hi) {
                if (tt > k && tt != exit && tt != n && !has(esc, tt)) return fallback(k + 1, hi, exit, esc, out);
                if (tt <= k && !has(esc, tt)) return fallback(k + 1, hi, exit, esc, out);
                if (k + 1 > hi) return tt != exit;
                if (!valid(k + 1, hi, exit, k, esc)) return fallback(k + 1, hi, exit, esc, out);
                Node i = guard(Node::If, k + 1);
                bool e = seq(k + 1, hi, exit, esc, i.body);
                out.push_back(std::move(i));
                return e || tt != exit;
            }
            std::vector<uint32_t> sub = esc;
            sub.push_back(exit);
            const Term& last = terms[tt - 1];
            if (last.kind == Exit::Jump && last.t > tt && (last.t <= hi || last.t == exit)) {
                uint32_t j = last.t;
                if (valid(k + 1, tt - 1, j, k, sub) && valid(tt, j - 1, j, k, sub)) {
                    Node i = guard(Node::IfElse, k + 1);
                    bool a = seq(k + 1, tt - 1, j, sub, i.body);
                    bool b = seq(tt, j - 1, j, sub, i.other);
                    out.push_back(std::move(i));
                    if (j > hi) return a || b;
                    if (a || b) return rest(j, hi, exit, esc, out);
                    k = j;
                    continue;
                }
            }
            if (!valid(k + 1, tt - 1, tt, k, sub)) return fallback(k + 1, hi, exit, esc, out);
            Node i = guard(Node::If, k + 1);
            bool a = seq(k + 1, tt - 1, tt, sub, i.body);
            out.push_back(std::move(i));
            if (a) return rest(tt, hi, exit, esc, out);
            k = tt;
        }
        return false;
    }
};

}  // namespace

const char* build_stamp() { return __DATE__ " " __TIME__; }

std::vector<Node> plan(const std::vector<Term>& terms, const std::vector<Region>& regions, std::string* why) {
    uint32_t n = uint32_t(terms.size());
    std::vector<Node> out;
    if (!n) return out;
    Planner p(terms, regions);
    if (!p.valid(0, n - 1, n, ANY, {})) {
        p.note("%s", "the program's branches do not nest");
        out.push_back({Node::Range, 0, n - 1});
    } else {
        p.seq(0, n - 1, n, {}, out);
    }
    if (why) *why = p.why;
    return out;
}

}  // namespace kl::shader::cfg

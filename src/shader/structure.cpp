#include "shader/structure.h"

#include <algorithm>

namespace kl::shader::cfg {
namespace {

constexpr uint32_t ANY = ~0u;

struct Planner {
    const std::vector<Term>& terms;
    const std::vector<Region>& regions;
    uint32_t n;
    std::vector<std::vector<uint32_t>> preds;
    std::vector<char> active;

    Planner(const std::vector<Term>& t, const std::vector<Region>& r) : terms(t), regions(r), n(uint32_t(t.size())) {
        preds.resize(n + 1);
        active.resize(r.size());
        for (uint32_t k = 0; k < n; ++k)
            for (uint32_t s : succ(k)) preds[std::min(s, n)].push_back(k);
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

    bool entered_mid_loop() const {
        for (auto& r : regions)
            for (uint32_t k = r.lo + 1; k <= r.hi && k < n; ++k)
                for (uint32_t p : preds[k])
                    if (p < r.lo || p > r.hi) return true;
        return false;
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
        Node g{Node::If, k};
        seq(k, hi, exit, esc, g.body);
        out.push_back(std::move(g));
        return true;
    }

    bool fallback(uint32_t k, uint32_t hi, std::vector<Node>& out) {
        if (k <= hi) out.push_back({Node::Range, k, hi});
        return true;
    }

    bool seq(uint32_t lo, uint32_t hi, uint32_t exit, const std::vector<uint32_t>& esc, std::vector<Node>& out) {
        for (uint32_t k = lo; k <= hi;) {
            if (int ri = region_at(k, hi); ri >= 0) {
                const Region& r = regions[size_t(ri)];
                std::vector<uint32_t> out_ok = esc;
                out_ok.push_back(exit);
                for (uint32_t x = r.lo; x <= r.hi; ++x)
                    for (uint32_t s : succ(x))
                        if ((s < r.lo || s > r.hi) && s != r.hi + 1 && s != n && !has(out_ok, s)) return fallback(k, hi, out);
                Node l{Node::Loop, uint32_t(ri)};
                active[size_t(ri)] = 1;
                std::vector<uint32_t> inner = out_ok;
                inner.push_back(r.lo);
                if (valid(r.lo, r.hi, r.hi + 1, ANY, inner)) seq(r.lo, r.hi, r.hi + 1, inner, l.body);
                else l.body.push_back({Node::Range, r.lo, r.hi});
                active[size_t(ri)] = 0;
                out.push_back(std::move(l));
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
            if (t.f != k + 1) return fallback(k + 1, hi, out);
            uint32_t tt = std::min(t.t, n);
            if (tt == k + 1) {
                ++k;
                continue;
            }
            if (tt <= k || tt > hi) {
                if (tt > k && tt != exit && tt != n && !has(esc, tt)) return fallback(k + 1, hi, out);
                if (tt <= k && !has(esc, tt)) return fallback(k + 1, hi, out);
                if (k + 1 > hi) return tt != exit;
                if (!valid(k + 1, hi, exit, k, esc)) return fallback(k + 1, hi, out);
                Node i{Node::If, k + 1};
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
                    Node i{Node::IfElse, k + 1};
                    bool a = seq(k + 1, tt - 1, j, sub, i.body);
                    bool b = seq(tt, j - 1, j, sub, i.other);
                    out.push_back(std::move(i));
                    if (j > hi) return a || b;
                    if (a || b) return rest(j, hi, exit, esc, out);
                    k = j;
                    continue;
                }
            }
            if (!valid(k + 1, tt - 1, tt, k, sub)) return fallback(k + 1, hi, out);
            Node i{Node::If, k + 1};
            bool a = seq(k + 1, tt - 1, tt, sub, i.body);
            out.push_back(std::move(i));
            if (a) return rest(tt, hi, exit, esc, out);
            k = tt;
        }
        return false;
    }
};

}  // namespace

std::vector<Node> plan(const std::vector<Term>& terms, const std::vector<Region>& regions) {
    uint32_t n = uint32_t(terms.size());
    std::vector<Node> out;
    if (!n) return out;
    Planner p(terms, regions);
    if (p.entered_mid_loop() || !p.valid(0, n - 1, n, ANY, {})) {
        out.push_back({Node::Range, 0, n - 1});
        return out;
    }
    p.seq(0, n - 1, n, {}, out);
    return out;
}

}  // namespace kl::shader::cfg

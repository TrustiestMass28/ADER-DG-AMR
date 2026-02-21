#include "AmrDG.h"

// --- Legendre<P> constructor ---

template<int P>
AmrDG::Legendre<P>::Legendre(amrex::Real x) {
    val[0] = 1.0; dval[0] = 0.0; ddval[0] = 0.0;
    if constexpr (P >= 1) {
        val[1] = x; dval[1] = 1.0; ddval[1] = 0.0;
    }
    for (int n = 2; n <= P; ++n) {
        val[n]  = ((2*n-1)*x*val[n-1] - (n-1)*val[n-2]) / n;
        dval[n] = n*val[n-1] + x*dval[n-1];
        ddval[n] = (n+1)*dval[n-1] + x*ddval[n-1];
    }
}

// --- BasisLegendre<P> method definitions ---

template<int P>
amrex::Real AmrDG::BasisLegendre<P>::phi_s(int idx, const amrex::Vector<amrex::Real>& x) {
    const auto& mi = MultiIndex<P, AMREX_SPACEDIM>::table[idx];
    amrex::Real result = 1.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        Legendre<P> leg(x[d]);
        result *= leg.val[mi.idx[d]];
    }
    return result;
}

template<int P>
amrex::Real AmrDG::BasisLegendre<P>::dphi_s(int idx, const amrex::Vector<amrex::Real>& x, int d) {
    const auto& mi = MultiIndex<P, AMREX_SPACEDIM>::table[idx];
    amrex::Real result = 1.0;
    for (int a = 0; a < AMREX_SPACEDIM; ++a) {
        Legendre<P> leg(x[a]);
        if (a != d) {
            result *= leg.val[mi.idx[a]];
        } else {
            result *= leg.dval[mi.idx[a]];
        }
    }
    return result;
}

template<int P>
amrex::Real AmrDG::BasisLegendre<P>::ddphi_s(int idx, const amrex::Vector<amrex::Real>& x, int d1, int d2) {
    const auto& mi = MultiIndex<P, AMREX_SPACEDIM>::table[idx];
    amrex::Real result = 1.0;
    for (int a = 0; a < AMREX_SPACEDIM; ++a) {
        Legendre<P> leg(x[a]);
        if (d1 == d2) {
            result *= (a != d1) ? leg.val[mi.idx[a]] : leg.ddval[mi.idx[a]];
        } else {
            if (a == d1 || a == d2) {
                result *= leg.dval[mi.idx[a]];
            } else {
                result *= leg.val[mi.idx[a]];
            }
        }
    }
    return result;
}

template<int P>
amrex::Real AmrDG::BasisLegendre<P>::phi_t(int idx, amrex::Real tau) {
    const auto& mi = MultiIndex<P, AMREX_SPACEDIM + 1>::table[idx];
    Legendre<P> leg(tau);
    return leg.val[mi.idx[AMREX_SPACEDIM]];
}

template<int P>
amrex::Real AmrDG::BasisLegendre<P>::dtphi_t(int idx, amrex::Real tau) {
    const auto& mi = MultiIndex<P, AMREX_SPACEDIM + 1>::table[idx];
    Legendre<P> leg(tau);
    return leg.dval[mi.idx[AMREX_SPACEDIM]];
}

template<int P>
amrex::Real AmrDG::BasisLegendre<P>::phi_st(int idx, const amrex::Vector<amrex::Real>& x) {
    const auto& mi = MultiIndex<P, AMREX_SPACEDIM + 1>::table[idx];
    amrex::Real result = 1.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        Legendre<P> leg(x[d]);
        result *= leg.val[mi.idx[d]];
    }
    Legendre<P> leg_t(x[AMREX_SPACEDIM]);
    result *= leg_t.val[mi.idx[AMREX_SPACEDIM]];
    return result;
}

template<int P>
auto AmrDG::BasisLegendre<P>::phi_s_all(const amrex::Vector<amrex::Real>& x)
    -> std::array<amrex::Real, Np_s>
{
    std::array<std::array<amrex::Real, P+1>, AMREX_SPACEDIM> leg_val;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        Legendre<P> leg(x[d]);
        leg_val[d] = leg.val;
    }
    std::array<amrex::Real, Np_s> result;
    for (int n = 0; n < Np_s; ++n) {
        const auto& mi = MultiIndex<P, AMREX_SPACEDIM>::table[n];
        amrex::Real prod = 1.0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            prod *= leg_val[d][mi.idx[d]];
        }
        result[n] = prod;
    }
    return result;
}

template<int P>
auto AmrDG::BasisLegendre<P>::phi_st_all(const amrex::Vector<amrex::Real>& x)
    -> std::array<amrex::Real, Np_st>
{
    std::array<std::array<amrex::Real, P+1>, AMREX_SPACEDIM + 1> leg_val;
    for (int d = 0; d < AMREX_SPACEDIM + 1; ++d) {
        Legendre<P> leg(x[d]);
        leg_val[d] = leg.val;
    }
    std::array<amrex::Real, Np_st> result;
    for (int n = 0; n < Np_st; ++n) {
        const auto& mi = MultiIndex<P, AMREX_SPACEDIM + 1>::table[n];
        amrex::Real prod = 1.0;
        for (int d = 0; d < AMREX_SPACEDIM + 1; ++d) {
            prod *= leg_val[d][mi.idx[d]];
        }
        result[n] = prod;
    }
    return result;
}

template<int P>
amrex::Vector<amrex::Vector<int>> AmrDG::BasisLegendre<P>::get_basis_idx_s() {
    constexpr auto& tab = MultiIndex<P, AMREX_SPACEDIM>::table;
    amrex::Vector<amrex::Vector<int>> result(Np_s, amrex::Vector<int>(AMREX_SPACEDIM));
    for (int n = 0; n < Np_s; ++n) {
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            result[n][d] = tab[n].idx[d];
        }
    }
    return result;
}

template<int P>
amrex::Vector<amrex::Vector<int>> AmrDG::BasisLegendre<P>::get_basis_idx_st() {
    constexpr auto& tab = MultiIndex<P, AMREX_SPACEDIM + 1>::table;
    amrex::Vector<amrex::Vector<int>> result(Np_st, amrex::Vector<int>(AMREX_SPACEDIM + 1));
    for (int n = 0; n < Np_st; ++n) {
        for (int d = 0; d < AMREX_SPACEDIM + 1; ++d) {
            result[n][d] = tab[n].idx[d];
        }
    }
    return result;
}

template<int P>
amrex::Vector<amrex::Vector<int>> AmrDG::BasisLegendre<P>::get_basis_idx_t() {
    constexpr auto& tab = MultiIndex<P, AMREX_SPACEDIM + 1>::table;
    amrex::Vector<amrex::Vector<int>> result(Np_st, amrex::Vector<int>(1));
    for (int n = 0; n < Np_st; ++n) {
        result[n][0] = tab[n].idx[AMREX_SPACEDIM];
    }
    return result;
}

// --- Explicit instantiations ---

template struct AmrDG::Legendre<1>;
template struct AmrDG::Legendre<2>;
template struct AmrDG::Legendre<3>;
template struct AmrDG::Legendre<4>;
template struct AmrDG::Legendre<5>;
template struct AmrDG::Legendre<6>;
template struct AmrDG::Legendre<7>;
template struct AmrDG::Legendre<8>;
template struct AmrDG::Legendre<9>;
template struct AmrDG::Legendre<10>;

template struct AmrDG::BasisLegendre<1>;
template struct AmrDG::BasisLegendre<2>;
template struct AmrDG::BasisLegendre<3>;
template struct AmrDG::BasisLegendre<4>;
template struct AmrDG::BasisLegendre<5>;
template struct AmrDG::BasisLegendre<6>;
template struct AmrDG::BasisLegendre<7>;
template struct AmrDG::BasisLegendre<8>;
template struct AmrDG::BasisLegendre<9>;
template struct AmrDG::BasisLegendre<10>;

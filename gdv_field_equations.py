import sympy as sp

# ============================================================
# Coordinates
# ============================================================

t, r, th, ph = sp.symbols("t r theta phi", real=True)
coords = [t, r, th, ph]



# ============================================================
# Functions and Symbols
# ============================================================
c0, c1, c2, c3 = sp.symbols("c0 c1 c2 c3", real=True)
rho = sp.symbols("rho", real=True)
p = sp.symbols("p", real=True)
a = sp.Function("a")(t)
c = sp.Integer(1)
b = sp.Function("b")(r)


adot = sp.diff(a, t)
br = sp.diff(b, r)
X = sp.simplify(adot - b)

# ============================================================
# Tetrad e^a_mu
# ============================================================

eta = sp.diag(-1, 1, 1, 1)

e = sp.zeros(4, 4)

e[0, 0] = c
e[1, 0] = r*b
e[1, 1] = a
e[2, 2] = r*a
e[3, 3] = r*a*sp.sin(th)

e_inv = sp.simplify(e.inv())

# ============================================================
# Metric
# ============================================================

g = sp.simplify(e.T * eta * e)
g_inv = sp.simplify(g.inv())

# ============================================================
# Spin connection omega^a_{b mu}
# ============================================================

omega = [[[
    sp.Integer(0)
    for mu in range(4)]
    for B in range(4)]
    for A in range(4)
]

omega[1][2][2] = -1
omega[2][1][2] = 1

omega[1][3][3] = -sp.sin(th)
omega[3][1][3] = sp.sin(th)

omega[2][3][3] = -sp.cos(th)
omega[3][2][3] = sp.cos(th)

# ============================================================
# Lorentz torsion T^a_{mu nu}
# ============================================================

T_lor = [[[
    sp.simplify(
        sp.diff(e[A, nu], coords[mu])
        - sp.diff(e[A, mu], coords[nu])
        + sum(omega[A][B][mu] * e[B, nu] for B in range(4))
        - sum(omega[A][B][nu] * e[B, mu] for B in range(4))
    )
    for nu in range(4)]
    for mu in range(4)]
    for A in range(4)
]

# ============================================================
# Coordinate torsion T^rhoo_{mu nu}
# ============================================================

T_up_down_down = [[[
    sp.simplify(
        sum(e_inv[rhoo, A] * T_lor[A][mu][nu] for A in range(4))
    )
    for nu in range(4)]
    for mu in range(4)]
    for rhoo in range(4)
]

# ============================================================
# Fully lowered torsion T_{rhoo mu nu}
# ============================================================

T_down_down_down = [[[
    sp.simplify(
        sum(g[rhoo, sig] * T_up_down_down[sig][mu][nu] for sig in range(4))
    )
    for nu in range(4)]
    for mu in range(4)]
    for rhoo in range(4)
]

# ============================================================
# Fully raised torsion T^{rhoo mu nu}
# ============================================================

T_up_up_up = [[[
    sp.simplify(
        sum(
            g_inv[rhoo, sig]
            * g_inv[mu, alpha]
            * g_inv[nu, betaa]
            * T_down_down_down[sig][alpha][betaa]
            for sig in range(4)
            for alpha in range(4)
            for betaa in range(4)
        )
    )
    for nu in range(4)]
    for mu in range(4)]
    for rhoo in range(4)
]

# ============================================================
# Torsion vector V_mu = T^rhoo_{rhoo mu}
# ============================================================

V_cov = [
    sp.simplify(
        sum(T_up_down_down[rhoo][rhoo][mu] for rhoo in range(4))
    )
    for mu in range(4)
]

V_contra = [
    sp.simplify(
        sum(g_inv[mu, nu] * V_cov[nu] for nu in range(4))
    )
    for mu in range(4)
]

# ============================================================
# Torsion scalar
#
# T = 1/4 T^{rhoo mu nu} T_{rhoo mu nu}
#   + 1/2 T^{rhoo mu nu} T_{nu mu rhoo}
#   - T^rhoo_{rhoo mu} T^nu_{nu}^{ mu}
# ============================================================

term1 = sp.simplify(
    sp.Rational(1, 4)
    * sum(
        T_up_up_up[rhoo][mu][nu]
        * T_down_down_down[rhoo][mu][nu]
        for rhoo in range(4)
        for mu in range(4)
        for nu in range(4)
    )
)

term2 = sp.simplify(
    sp.Rational(1, 2)
    * sum(
        T_up_up_up[rhoo][mu][nu]
        * T_down_down_down[nu][mu][rhoo]
        for rhoo in range(4)
        for mu in range(4)
        for nu in range(4)
    )
)

term3 = sp.simplify(
    -sum(V_cov[mu] * V_contra[mu] for mu in range(4))
)

Tscalar = sp.simplify(sp.factor(term1 + term2 + term3))

# ============================================================
# Expected result
# ============================================================

T_expected = sp.simplify(
    2 * X * (3*X - 2*r*br) / (a**2 * c**2)
)

check = sp.simplify(sp.factor(Tscalar - T_expected))

# ============================================================
# FLRW limit
# ============================================================

T_FLRW = sp.simplify(
    Tscalar.subs({
        b: 0,
        br: 0,
        c: 1
    })
)



# ============================================================
# T^{mu nu}_{ rhoo}
# ============================================================

T_up_up_down = [[[
    sp.simplify(
        sum(
            g_inv[mu, alpha]
            * g_inv[nu, betaa]
            * T_down_down_down[alpha][betaa][rhoo]
            for alpha in range(4)
            for betaa in range(4)
        )
    )
    for rhoo in range(4)]
    for nu in range(4)]
    for mu in range(4)
]

# ============================================================
# T_{rhoo}^{ mu nu}
# ============================================================

T_down_up_up = [[[
    sp.simplify(
        sum(
            g_inv[mu, alpha]
            * g_inv[nu, betaa]
            * T_down_down_down[rhoo][alpha][betaa]
            for alpha in range(4)
            for betaa in range(4)
        )
    )
    for nu in range(4)]
    for mu in range(4)]
    for rhoo in range(4)
]

# ============================================================
# Kontorsiyon K^{mu nu}_{ rhoo}
#
# K^{mu nu}_{ rhoo}
# =
# -1/2 ( T^{mu nu}_{ rhoo}
#       -T^{nu mu}_{ rhoo}
#       -T_{rhoo}^{ mu nu} )
# ============================================================

K_up_up_down = [[[
    sp.simplify(
        -sp.Rational(1, 2) *
        (
            T_up_up_down[mu][nu][rhoo]
            - T_up_up_down[nu][mu][rhoo]
            - T_down_up_up[rhoo][mu][nu]
        )
    )
    for rhoo in range(4)]
    for nu in range(4)]
    for mu in range(4)
]

# ============================================================
# Trace Q^nu = T^{alpha nu}_{ alpha}
# ============================================================

Q_contra = [
    sp.simplify(
        sum(T_up_up_down[alpha][nu][alpha] for alpha in range(4))
    )
    for nu in range(4)
]

# ============================================================
# Superpotential S_rhoo^{mu nu}
#
# S_rhoo^{mu nu}
# =
# 1/2 ( K^{mu nu}_{ rhoo}
#      +delta^mu_rhoo T^{alpha nu}_{ alpha}
#      -delta^nu_rhoo T^{alpha mu}_{ alpha} )
# ============================================================

S_down_up_up = [[[
    sp.simplify(
        sp.Rational(1, 2) *
        (
            K_up_up_down[mu][nu][rhoo]
            + (1 if mu == rhoo else 0) * Q_contra[nu]
            - (1 if nu == rhoo else 0) * Q_contra[mu]
        )
    )
    for nu in range(4)]
    for mu in range(4)]
    for rhoo in range(4)
]

# ============================================================
# Check: T = S_rhoo^{mu nu} T^rhoo_{mu nu}
# ============================================================

T_from_S = sp.simplify(
    sp.factor(
        sum(
            S_down_up_up[rhoo][mu][nu] * T_up_down_down[rhoo][mu][nu]
            for rhoo in range(4)
            for mu in range(4)
            for nu in range(4)
        )
    )
)

check_S = sp.simplify(sp.factor(T_from_S - Tscalar))



# ============================================================
# 0. PARAMETERS
# ============================================================

beta, kappa = sp.symbols("beta kappa")

det_e = sp.simplify(e.det())

# ============================================================
# 1. S_a^{mu nu} = E_a^rhoo S_rhoo^{mu nu}
# ============================================================

S_lor = [[[
    sp.simplify(
        sum(e_inv[rhoo, A] * S_down_up_up[rhoo][mu][nu] for rhoo in range(4))
    )
    for nu in range(4)]
    for mu in range(4)]
    for A in range(4)
]

# ============================================================
# 2. Lorentz-covariant density divergence
#
# e^{-1} D_mu(e S_a^{mu nu})
#
# lower Lorentz index:
# D_mu X_a = partial_mu X_a - omega^b_{a mu} X_b
# ============================================================

def div_density_S(A, nu):

    result = 0

    for mu in range(4):

        partial_part = sp.diff(
            det_e * S_lor[A][mu][nu],
            coords[mu]
        )

        spin_part = sum(
            omega[B][A][mu] * det_e * S_lor[B][mu][nu]
            for B in range(4)
        )

        result += partial_part - spin_part

    return sp.simplify(result / det_e)

# ============================================================
# 3. TEGR tensor G_a^nu
#
# IMPORTANT:
# With our S convention the torsion term has PLUS sign:
#
# G_a^nu =
# e^{-1} D_mu(e S_a^{mu nu})
# + E_a^lambda T^rhoo_{mu lambda} S_rhoo^{nu mu}
# - 1/4 E_a^nu T
# ============================================================

G = [[
    sp.factor(
        div_density_S(A, nu)
        + sum(
            e_inv[lam, A]
            * T_up_down_down[rho][lam][mu]
            * S_down_up_up[rho][nu][mu]
            for lam in range(4)
            for rho in range(4)
            for mu in range(4)
        )
        - sp.Rational(1, 4) * e_inv[nu, A] * Tscalar
    )
    for nu in range(4)] for A in range(4)]

# ============================================================
# 4. T_a and T^a
# ============================================================

T_lor_down = [
    sp.simplify(
        sum(e_inv[mu, A] * V_cov[mu] for mu in range(4))
    )
    for A in range(4)
]

T_lor_up = [
    sp.simplify(
        sum(e[A, mu] * V_contra[mu] for mu in range(4))
    )
    for A in range(4)
]

T2_scalar = sp.simplify(
    sum(V_cov[mu] * V_contra[mu] for mu in range(4))
)

# ============================================================
# 5. A_a^{nu rhoo}
#
# A_a^{nu rhoo}
# =
# 2e(T^rhoo E_a^nu - T^nu E_a^rhoo)
# ============================================================

A_tensor = [[[
    sp.simplify(
        2 * det_e *
        (
            V_contra[rhoo] * e_inv[nu, A]
            -
            V_contra[nu] * e_inv[rhoo, A]
        )
    )
    for rhoo in range(4)]
    for nu in range(4)]
    for A in range(4)
]

# ============================================================
# 6. D_nu A_a^{nu rhoo}
# ============================================================

def D_A(A, rhoo):

    result = 0

    for nu in range(4):

        partial_part = sp.diff(
            A_tensor[A][nu][rhoo],
            coords[nu]
        )

        spin_part = sum(
            omega[B][A][nu] * A_tensor[B][nu][rhoo]
            for B in range(4)
        )

        result += partial_part - spin_part

    return sp.simplify(result)

# ============================================================
# 7. Beta-sector tensor V_a^rhoo
#
# V_a^rhoo =
# T^2 E_a^rhoo
# - 2 T^rhoo T_a
# - 2 T^mu E_a^nu T^rhoo_{nu mu}
# - e^{-1}D_nu[2e(T^rhoo E_a^nu - T^nu E_a^rhoo)]
# ============================================================

V = [[
    sp.simplify(
        T2_scalar * e_inv[rhoo, A]
        -
        2 * V_contra[rhoo] * T_lor_down[A]
        -
        2 * sum(
            V_contra[mu]
            * e_inv[nu, A]
            * T_up_down_down[rhoo][nu][mu]
            for mu in range(4)
            for nu in range(4)
        )
        -
        D_A(A, rhoo) / det_e
    )
    for rhoo in range(4)]
    for A in range(4)
]

# ============================================================
# 8. Full LHS
#
# G_a^nu + beta/4 V_a^nu
# ============================================================

LHS = [[
    sp.simplify(
        -G[A][nu] + beta * V[A][nu] / 4
    )
    for nu in range(4)]
    for A in range(4)
]
# ============================================================
# FIELD EQUATIONS
# ============================================================

FieldLHS = [[
    sp.simplify(sp.factor(LHS[A][nu]))
    for nu in range(4)]
    for A in range(4)
]




u_up=sp.zeros(1,4)
u_up[0]=1
u_up[1]=-b*r/a

ThetaSon_a_ust_nu=sp.simplify((rho + p )* g *e_inv*u_up.T * u_up + p * g * e_inv * g_inv)
RHS_a_nu = (kappa/2)*ThetaSon_a_ust_nu

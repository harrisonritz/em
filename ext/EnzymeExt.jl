module EMEnzymeExt

using EM
using Enzyme: Enzyme
using Optim: optimize, LBFGS

import EM: _optimizesubject, _autodiff_hessian

# Enzyme reverse-mode gradient for Optim's g!(G, x) interface.
# Accumulates into G in-place; fill! is required because Enzyme adds to the shadow.
function _optimizesubject(likfun, startx, ::Val{:enzyme})
    function g!(G, x)
        fill!(G, 0.0)
        Enzyme.autodiff(Enzyme.Reverse, likfun, Enzyme.Duplicated(x, G))
    end
    a = optimize(likfun, g!, startx, LBFGS())
    return (a.minimum, a.minimizer)
end

# Enzyme hessian via forward-over-reverse: O(n) cost for scalar f: ℝⁿ → ℝ,
# compared to ForwardDiff's forward-over-forward which is O(n²).
function _autodiff_hessian(f, x, ::Val{:enzyme})
    Enzyme.hessian(Enzyme.Reverse, f, x)
end

end

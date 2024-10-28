import jax
import jax.numpy as jnp


@jax.jit
def bispline_interp(xnew,ynew,xp,yp,zp):
    """
    (xnew,ynew): two 1D vector  of same size where to perform predictions  f(xnew[i],ynew[i])
    (xp,yp): original grid points 1D vector
    zp: original values of functions  zp[i,j] = value at xp[i], yp[j]
    """
    
    M = 1./16 * jnp.array([[0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                           [0, 0, 0, 0, -8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                           [0, 0, 0, 0, 16, -40, 32, -8, 0, 0, 0, 0, 0, 0, 0, 0], 
                           [0, 0, 0, 0, -8, 24, -24, 8, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, -8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0], 
                           [4, 0, -4, 0, 0, 0, 0, 0, -4, 0, 4, 0, 0, 0, 0, 0], 
                           [-8, 20, -16, 4, 0, 0, 0, 0, 8, -20, 16, -4, 0, 0, 0, 0],
                           [4, -12, 12, -4, 0, 0, 0, 0, -4, 12, -12, 4, 0, 0, 0, 0],
                           [0, 16, 0, 0, 0, -40, 0, 0, 0, 32, 0, 0, 0, -8, 0, 0], 
                           [-8, 0, 8, 0, 20, 0, -20, 0, -16, 0, 16, 0, 4, 0, -4, 0], 
                           [16, -40, 32, -8, -40, 100, -80, 20, 32, -80, 64, -16, -8, 20, -16, 4], 
                           [-8, 24, -24, 8, 20, -60, 60, -20, -16, 48, -48, 16, 4, -12, 12, -4], 
                           [0, -8, 0, 0, 0, 24, 0, 0, 0, -24, 0, 0, 0, 8, 0, 0], 
                           [4, 0, -4, 0, -12, 0, 12, 0, 12, 0, -12, 0, -4, 0, 4, 0], 
                           [-8, 20, -16, 4, 24, -60, 48, -12, -24, 60, -48, 12, 8, -20, 16, -4], 
                           [4, -12, 12, -4, -12, 36, -36, 12, 12, -36, 36, -12, -4, 12, -12, 4]]
                        )
    
    M1 = jnp.array([[1.,0.,0.,0.],
                    [-1.,1.,0.,0.],
                    [-1.,0.,1.,0.],
                    [1.,-1.,-1.,1.]])

    def built_Ivec(zp,ix,iy):
        return jnp.array([zp[ix+i,iy+j] for j in range(-1,3) for i in range(-1,3)])


    def built_Ivec1(zp,ix,iy):
        return jnp.array([zp[ix+i,iy+j] for j in range(0,2) for i in range(0,2)])

    
    
    def compute_basis(x,order=3):
        """
        x in [0,1]
        """ 
        return jnp.array([x**i for i in jnp.arange(0, order+1)])
    
    def tval(xnew,ix,xp):
        return (xnew-xp[ix-1])/(xp[ix]-xp[ix-1])
    
    ix = jnp.clip(jnp.searchsorted(xp, xnew, side="right"), 0, len(xp)-1)
    iy = jnp.clip(jnp.searchsorted(yp, ynew, side="right"), 0, len(yp)-1)

    def bilinear_interp(ix,iy):
        Iv = built_Ivec1(zp,ix-1,iy-1)
        av = M1 @ Iv
        amtx = av.reshape(2,2,-1)
        tx = tval(xnew,ix,xp)
        ty = tval(ynew,iy,yp)
        basis_x = compute_basis(tx,order=1)
        basis_y = compute_basis(ty,order=1)
        res = jnp.einsum("i...,ij...,j...",basis_y,amtx,basis_x)
        return res

    def bispline_interp(ix,iy):
        Iv = built_Ivec(zp,ix-1,iy-1)
        av = M @ Iv
        amtx = av.reshape(4,4,-1)
        tx = tval(xnew,ix,xp)
        ty = tval(ynew,iy,yp)
        basis_x = compute_basis(tx)
        basis_y = compute_basis(ty)
        res = jnp.einsum("i...,ij...,j...",basis_y,amtx,basis_x)
        return res
    
    condx = jnp.logical_and(ix>=2, ix<=len(xp)-2)
    condy = jnp.logical_and(iy>=2, iy<=len(yp)-2)
    
    cond = jnp.logical_and(condx,condy)
    return jnp.where(cond,
             bispline_interp(ix,iy),
             bilinear_interp(ix,iy))
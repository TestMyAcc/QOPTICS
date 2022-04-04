C FILE: LAPLACIAN.F
      subroutine del2_loop(F, del2F, n, m, p, dx, dy, dz)

      INTEGER :: n,m,p
      real(kind=8) :: F(n,m,p)
      real(kind=8)  :: del2F(n,m,p)
      real(kind=8) :: dx, dy, dz
      INTEGER i,j,k
cf2py intent(in) :: F
cf2py intent(out) :: del2F
cf2py intent(hide) :: n,m,p

      do i = 2, m-1
         do j = 2, n-1
            do k = 2, p-1
               del2F(i,j,k) =  dx**(-2)*(F(i+1,j, k)-2*F(i,j,k)
     1            + F(i-1,j,k)) 
     2            + dy**(-2)*( F(i,j+1,k)-2*F(i,j,k)
     3            + F(i,j-1,k) ) 
     4            + dz**(-2)*( F(i,j,k+1)-2*F(i,j,k)+F(i,j,k-1) )
               enddo
         enddo
      enddo

      end subroutine del2_loop


      subroutine del2_ele(F_, ddel2F_, n_, m_, p_, dx_, dy_, dz_)
      INTEGER :: n_,m_,p_
      real(kind=8) :: F_(n_,m_,p_)
      real(kind=8)  :: ddel2F_(n_,m_,p_)
      real(kind=8) :: dx_, dy_, dz_
cf2py intent(in) :: F_
cf2py intent(out) :: ddel2F_
cf2py intent(hide) :: n_,m_,p_
  
     
      ddel2F_(2:n_-1,2:m_-1,2:p_-1) = (
     1      (1/dx_**2)*(
     2            F_(3:n_,   2:m_-1, 2:p_-1) 
     3        - 2*F_(2:n_-1, 2:m_-1, 2:p_-1) 
     4          + F_(1:n_-2, 2:m_-1, 2:p_-1))
     5      +(1/dy_**2)*(
     6            F_(2:n_-1, 3:m_,   2:p_-1) 
     7        - 2*F_(2:n_-1, 2:m_-1, 2:p_-1) 
     8          + F_(2:n_-1, 1:m_-2, 2:p_-1))
     9      +(1/dz_**2)*(
     a            F_(2:n_-1, 2:m_-1, 3:p_) 
     b        - 2*F_(2:n_-1, 2:m_-1, 2:p_-1) 
     c          + F_(2:n_-1, 2:m_-1, 1:p_-2)))
  
        
    !   F_(2:9,2:9,2:9) = F(2:9, 2:9, 2:9) 
  
      end subroutine del2_ele
     
      subroutine del2_concurloop(
     1       F__,del2F__, n__, m__, p__, dx__, dy__, dz__) 
      INTEGER :: n__,m__,p__
      real(kind=8) :: F__(n__,m__,p__)
      real(kind=8)  :: del2F__(n__,m__,p__)
      real(kind=8) :: dx__, dy__, dz__
      INTEGER i__,j__,k__
cf2py intent(in) :: F__
cf2py intent(out) :: del2F__
cf2py intent(hide) :: n__,m__,p__

      end subroutine del2_concurloop


      subroutine del2_mtx(
     1       F___,del2F___, n___, m___, p___, dx___, dy___, dz___) 
      INTEGER :: n___,m___,p___
      real(kind=8) :: F___(n___,m___,p___)
      real(kind=8)  :: del2F___(n___,m___,p___)
      real(kind=8) :: dx___, dy___, dz___
      INTEGER i___,j___,k___
cf2py intent(in) :: F___
cf2py intent(out) :: del2F___
cf2py intent(hide) :: n___,m___,p___


      end subroutine del2_mtx

C END FILE LAPLACIAN.F
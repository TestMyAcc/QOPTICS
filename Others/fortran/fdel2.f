C FILE: LAPLACIAN.F
C complied in unix env
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
               del2F(i,j,k) =  dy**(-2)*(F(i+1,j, k)-2*F(i,j,k)
     1            + F(i-1,j,k)) 
     2            + dx**(-2)*( F(i,j+1,k)-2*F(i,j,k)
     3            + F(i,j-1,k) ) 
     4            + dz**(-2)*( F(i,j,k+1)-2*F(i,j,k)+F(i,j,k-1) )
               enddo
         enddo
      enddo

      end subroutine del2_loop


      subroutine del2_ele(F_, del2F_, n_, m_, p_, dx_, dy_, dz_)
      INTEGER :: n_,m_,p_
      real(kind=8) :: F_(n_,m_,p_)
      real(kind=8)  :: del2F_(n_,m_,p_)
      real(kind=8) :: dx_, dy_, dz_
cf2py intent(in) :: F_
cf2py intent(out) :: del2F_
cf2py intent(hide) :: n_,m_,p_
  
     
      del2F_(2:n_-1,2:m_-1,2:p_-1) = (
     1      (1/dy_**2)*(
     2            F_(3:n_,   2:m_-1, 2:p_-1) 
     3        - 2*F_(2:n_-1, 2:m_-1, 2:p_-1) 
     4          + F_(1:n_-2, 2:m_-1, 2:p_-1))
     5      +(1/dx_**2)*(
     6            F_(2:n_-1, 3:m_,   2:p_-1) 
     7        - 2*F_(2:n_-1, 2:m_-1, 2:p_-1) 
     8          + F_(2:n_-1, 1:m_-2, 2:p_-1))
     9      +(1/dz_**2)*(
     a            F_(2:n_-1, 2:m_-1, 3:p_) 
     b        - 2*F_(2:n_-1, 2:m_-1, 2:p_-1) 
     c          + F_(2:n_-1, 2:m_-1, 1:p_-2)))
  
      end subroutine del2_ele
     
C END FILE LAPLACIAN.F
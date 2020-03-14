    module compositionvar
    use meshelement
    real(dp) ::   Lx=300.0 ! unit= A
    real(dp) ::Ly=300.0 ! chip size= 30nm*30nm
    real(dp) ::Lqw=30.0 ! Quantum well thickness= 3.0nm
    real(dp) ::Lcap=20.0 ! AlGaN cap layer thickness= 2.0nm
    real(dp) ::Lba=80.0 ! barrier thickness= 10.0nm
    real(dp) ::Lebl=100.0
    integer(i8b) ::qwnum=5
    integer(i8b) ::capnum=5
    integer(i8b) ::eblnum=1
    real(dp) ::Incomp_qw=0.15
    real(dp) ::Alcomp_cap=0.00
    real(dp) ::Alcomp_ebl=0.10
    real(dp) ::L_nGaN =2000.0
    ! ^ assigned value of device mesh and material ^

    integer(i8b) ::largeW=18 ! better > 16
    ! ^ random composition process ^

    real(dp) ::sigma=2.0
    real(dp) ::gwindow=3.0
    end module

    subroutine compositionmapgen
    use meshelement
    use global3d
    use compositionvar
    implicit none
    real(dp) :: x,y,z,xp,yp,zp,dd,cc
    integer(i8b) :: nodetype ,i ,j, k , ii, jj, kk, iii, jjj, kkk, test, pp
    !
    real,parameter :: me_GaN_parall=0.21, me_GaN_perpen=0.2, mhh_GaN=1.87, mlh_GaN=0.14
    real,parameter :: me_InN_parall=0.07, me_InN_perpen=0.07, mhh_InN=1.61 , mlh_InN=0.11
    real,parameter :: me_AlN_parall=0.32, me_AlN_perpen=0.3, mhh_AlN=2.68, mlh_AlN=0.26
    real,parameter :: e=1.6e-19
    real           :: Psp_InGaN, Psp_GaN, Psp_AlGaN, Ppz_InGaN_GaN, Ppz_AlGaN_GaN
    ! ^ semiconductor parameters ^

    real*8 :: density, grid_x, grid_y, grid_z, lattice_parameter
    ! ^ GaN parameters based on previous calculations ^

    !real*8 :: Incomp_qw, Alcomp_cap, Alcomp_ebl, Lx, Ly, Lqw, Lcap, Lebl, Lba,L_nGaN
    integer(i8b) ::  qwmid, capmid, eblmid
    real*8,allocatable ::qw_begin(:), qw_end(:), cap_begin(:), cap_end(:), ebl_begin(:), ebl_end(:)
    ! ^ assigned value of device mesh and material ^

    integer(i8b) ::  Nx, Ny, Nqw, Ncap, Nebl
    real*8, allocatable :: r01qw(:,:,:,:), rmapqw(:,:,:,:), gmapqw(:,:,:,:)
    real*8, allocatable :: r01cap(:,:,:,:), rmapcap(:,:,:,:), gmapcap(:,:,:,:)
    real*8, allocatable :: r01ebl(:,:,:,:), rmapebl(:,:,:,:), gmapebl(:,:,:,:)
    ! ^ random composition process ^

    real*8 :: weight, total_weight, weighted_rmap, distance2, diffuse_length
    ! ^ Gaussian process ^

    real*8,allocatable ::rqwavgz(:,:), gqwavgz(:,:), rcapavgz(:,:), gcapavgz(:,:), reblavgz(:,:), geblavgz(:,:)
    real*8,allocatable ::zvalueqw(:,:), zvaluecap(:,:), zvalueebl(:,:)
    real*8 :: Incom_inter, Alcom_inter, Gacom_inter, u_inter, v_inter, w_inter, Egu, Egv, Egw
    ! ^ output process ^
    INTEGER seed(2)

    !--------------------------------------------------------------------------------------------------

    seed(1)=-11582406 ! seeding number, can be changed
    call random_seed(PUT=seed)


    density = 0.044 ! don't change
    lattice_parameter = density**(-1.0/3.0)
    grid_x=lattice_parameter
    grid_y=lattice_parameter
    grid_z=lattice_parameter ! 2.833A, ref. wikipedia Ga, 1 atom per cubic
    ! ^ GaN parameters based on previous calculations ^

    Lx=300.0 ! unit= A
    Ly=300.0 ! chip size= 30nm*30nm
    Lqw=30.0 ! Quantum well thickness= 3.0nm
    Lba=80.0 ! barrier thickness= 10.0n
    Lebl=100.0
    qwnum=1
    capnum=1
    eblnum=1
    Incomp_qw=0.22
    Alcomp_ebl=0.00
    Alcomp_cap=0.20
    L_nGaN =400.0
    Lcap=20.0
    ! ^ assigned value of device mesh and material ^

    largeW=18 ! better > 16
    ! ^ random composition process ^

    sigma=2.0
    gwindow=3.0
    ! ^ Gaussian process ^

    !---------------------------------------------------------------------------------------------

    allocate(qw_begin(qwnum),qw_end(qwnum))
    allocate(cap_begin(capnum),cap_end(capnum))
    allocate(ebl_begin(eblnum),ebl_end(eblnum))
    do i=1,qwnum
        if(i  .eq. 1) then
            qw_begin(1) = L_nGaN + Lba ! unit: A
            qw_end(1)   = qw_begin(1) + Lqw
            cap_begin(1)= qw_end(1)+10
            cap_end(1)  = cap_begin(1)+ Lcap
        else
            qw_begin(i) = qw_end(i-1)  + Lba + Lcap
            qw_end(i)   = qw_begin(i) + Lqw
            cap_begin(i)= qw_end(i)+10
            cap_end(i)  = cap_begin(i)+ Lcap
          
        end if

    end do

    ebl_begin(1)=  qw_end(qwnum)  + Lba
    ebl_end(1)  = ebl_begin(1)+ Lebl

    Nx=nint(Lx/grid_x)+1 ! rounds with the nearest whole number
    Ny=nint(Ly/grid_y)+1
    Nqw=nint(Lqw/grid_z)+largeW
    Ncap=nint(Lcap/grid_z)+largeW
    Nebl=nint(Lebl/grid_z)+largeW
    qwmid=int(Nqw/2)+1
    capmid=int(Ncap/2)+1
    eblmid=int(Nebl/2)+1

    allocate(r01qw(Nx,Ny,Nqw,qwnum),rmapqw(Nx,Ny,Nqw,qwnum),gmapqw(Nx,Ny,Nqw,qwnum))
    allocate(zvalueqw(Nqw,qwnum),rqwavgz(Nqw,qwnum),gqwavgz(Nqw,qwnum))

    allocate(r01cap(Nx,Ny,Ncap,capnum),rmapcap(Nx,Ny,Ncap,capnum),gmapcap(Nx,Ny,Ncap,capnum))
    allocate(zvaluecap(Ncap,capnum),rcapavgz(Ncap,capnum),gcapavgz(Ncap,capnum))

    allocate(r01ebl(Nx,Ny,Nebl,eblnum),rmapebl(Nx,Ny,Nebl,eblnum),gmapebl(Nx,Ny,Nebl,eblnum))
    allocate(zvalueebl(Nebl,eblnum),reblavgz(Nebl,eblnum),geblavgz(Nebl,eblnum))



    write(*,*) "original map="

    !!should not relate to msh
    do jj=1,qwnum
        do kk=1,Nqw
            zvalueqw(kk,jj)=qw_begin(jj)+(Lqw/2)+(kk-qwmid)*grid_z
        end do
    end do

    do jj=1,capnum
        do kk=1,Ncap
            zvaluecap(kk,jj)=cap_begin(jj)+(Lcap/2)+(kk-capmid)*grid_z
        end do
    end do

    do jj=1,eblnum
        do kk=1,Nebl
            zvalueebl(kk,jj)=ebl_begin(jj)+(Lebl/2)+(kk-eblmid)*grid_z
        end do
    end do

    call random_number(r01qw)
    call random_number(r01cap)
    call random_number(r01ebl)
    !!------------------------------------
    write(*,*) "generated random map="
    rmapqw=0.0
    gmapqw=0.0
    !! random process to assign In atoms
    do pp=1,qwnum
        do ii=1,Nx
            do jj=1,Ny
                do kk=(largeW/2)+4,(largeW/2)+nint(Lqw/grid_z)
                    if (r01qw(ii,jj,kk,pp)<Incomp_qw)then
                        rmapqw(ii,jj,kk,pp)=1.0
                    else
                        rmapqw(ii,jj,kk,pp)=0.0
                    end if
                end do
            end do
        end do
    end do
    
    rmapcap=0.0
    gmapcap=0.0
    !! random process to assign In atoms
    do pp=1,capnum
        do ii=1,Nx
            do jj=1,Ny
                do kk=(largeW/2)+1,(largeW/2)+nint(Lcap/grid_z)-1
                    if (r01cap(ii,jj,kk,pp)<Alcomp_cap)then
                        rmapcap(ii,jj,kk,pp)=1.0
                    else
                        rmapcap(ii,jj,kk,pp)=0.0
                    end if
                end do
            end do
        end do
    end do
    rmapebl=0.0
    gmapebl=0.0
    !! random process to assign In atoms
    do pp=1,eblnum
        do ii=1,Nx
            do jj=1,Ny
                do kk=(largeW/2)+1,(largeW/2)+nint(Lebl/grid_z)
                    if (r01ebl(ii,jj,kk,pp)<Alcomp_ebl)then
                        rmapebl(ii,jj,kk,pp)=1.0
                    else
                        rmapebl(ii,jj,kk,pp)=0.0
                    end if
                end do
            end do
        end do
    end do


    ! integrate In comp. of each layer
    rqwavgz=0.0
    do pp=1,qwnum
        do kk=1,Nqw
            rqwavgz(kk,pp)=sum(rmapqw(:,:,kk,pp))
        end do
    end do

    write(*,*)'avg_In_randommap'
    do pp=1, qwnum
        do kk=1,Nqw
            rqwavgz(kk,pp)=rqwavgz(kk,pp)/Nx/Ny
            write(*,*) zvalueqw(kk,pp),rqwavgz(kk,pp)
        end do
    end do


    reblavgz=0.0
    do pp=1,eblnum
        do kk=1,Nebl
            reblavgz(kk,pp)=sum(rmapebl(:,:,kk,pp))
        end do
    end do
    write(*,*)'avg_AlEBL_randommap'
    do pp=1, eblnum
        do kk=1,Nebl
            reblavgz(kk,pp)=reblavgz(kk,pp)/Nx/Ny
            write(*,*) zvalueebl(kk,pp),reblavgz(kk,pp)
        end do
    end do


    write(*,*) "Gaussian map="

    do pp=1,qwnum
        do ii=1,Nx
            do jj=1,Ny
                do kk=1,Nqw
                    weighted_rmap = 0.0
                    total_weight  = 0.0
                    do i = max(ii-nint(gwindow*sigma),1), min(ii+nint(gwindow*sigma),Nx)
                        do j = max(jj-nint(gwindow*sigma),1), min(jj+nint(gwindow*sigma),Ny)
                            do k = max(kk-nint(gwindow*sigma),1), min(kk+nint(gwindow*sigma),Nqw)
                                distance2     = real( (i-ii)**2 + (j-jj)**2 + (k-kk)**2 )
                                weight        = exp(-distance2/(2*sigma**2))
                                weighted_rmap = weighted_rmap + weight*rmapqw(i,j,k,pp)
                                total_weight = total_weight + weight
                            end do
                        end do
                    end do
                    gmapqw(ii,jj,kk,pp) = weighted_rmap/total_weight
                end do
            end do
        end do
        write(*,*)'QWnumber',pp
    end do

    do pp=1,capnum
        do ii=1,Nx
            do jj=1,Ny
                do kk=1,Ncap
                    weighted_rmap = 0.0
                    total_weight  = 0.0
                    do i = max(ii-nint(gwindow*sigma),1), min(ii+nint(gwindow*sigma),Nx)
                        do j = max(jj-nint(gwindow*sigma),1), min(jj+nint(gwindow*sigma),Ny)
                            do k = max(kk-nint(gwindow*sigma),1), min(kk+nint(gwindow*sigma),Ncap)
                                distance2     = real( (i-ii)**2 + (j-jj)**2 + (k-kk)**2 )
                                weight        = exp(-distance2/(2*sigma**2))
                                weighted_rmap = weighted_rmap + weight*rmapcap(i,j,k,pp)
                                total_weight = total_weight + weight
                            end do
                        end do
                    end do
                    gmapcap(ii,jj,kk,pp) = weighted_rmap/total_weight
                end do
            end do
        end do
        write(*,*)'capnumber',pp
    end do

    do pp=1,eblnum
        do ii=1,Nx
            do jj=1,Ny
                do kk=1,Nebl
                    weighted_rmap = 0.0
                    total_weight  = 0.0
                    do i = max(ii-nint(gwindow*sigma),1), min(ii+nint(gwindow*sigma),Nx)
                        do j = max(jj-nint(gwindow*sigma),1), min(jj+nint(gwindow*sigma),Ny)
                            do k = max(kk-nint(gwindow*sigma),1), min(kk+nint(gwindow*sigma),Nebl)
                                distance2     = real( (i-ii)**2 + (j-jj)**2 + (k-kk)**2 )
                                weight        = exp(-distance2/(2*sigma**2))
                                weighted_rmap = weighted_rmap + weight*rmapebl(i,j,k,pp)
                                total_weight = total_weight + weight
                            end do
                        end do
                    end do
                    gmapebl(ii,jj,kk,pp) = weighted_rmap/total_weight
                end do
            end do
        end do
        write(*,*)'EBLnumber',pp
    end do

    !do pp=1,qwnum
    !   do kk=(largeW/2)+nint(Lqw/grid_z)+1,Nqw
    !      gmapqw(:,:,kk,pp)=0
    !   end do
    !end do
    !do pp=1,capnum
    !   do kk=1,largeW/2
    !      gmapcap(:,:,kk,pp)=0
    !   end do
    !end do

    ! integrate In comp. of each layer
    gqwavgz=0.0
    do pp=1,qwnum
        do kk=1,Nqw
            gqwavgz(kk,pp)=sum(gmapqw(:,:,kk,pp))
            gqwavgz(kk,pp)=gqwavgz(kk,pp)/Nx/Ny
        end do
    end do

    open(158,FILE="QWInavg.out")
    do kk=1,Nqw
        write(158,999) zvalueqw(kk,1),gqwavgz(kk,:)
    end do
    close(158)

gcapavgz=0.0
    do pp=1,capnum
        do kk=1,Ncap
            gcapavgz(kk,pp)=sum(gmapcap(:,:,kk,pp))
            gcapavgz(kk,pp)=gcapavgz(kk,pp)/Nx/Ny
        end do
    end do

    open(159,FILE="capAlavg.out")
    do kk=1,Ncap
        write(159,999) zvaluecap(kk,1),gcapavgz(kk,:)
    end do
    close(159)


    geblavgz=0.0
    do pp=1,eblnum
        do kk=1,Nebl
            geblavgz(kk,pp)=sum(gmapebl(:,:,kk,pp))
            geblavgz(kk,pp)=geblavgz(kk,pp)/Nx/Ny
        end do
    end do

    open(157,FILE="EBLAlavg.out")
    do kk=1,Nebl
        write(157,999) zvalueebl(kk,1),geblavgz(kk,:)
    end do
    close(157)



    open(154,FILE="In_map.out")
    write(154,"(a11)")"$MeshFormat"
    write(154,"(a5)")"2 0 8"
    write(154,"(a9)")"$NodeData"
    write(154,"(a1)")"1"
    write(154,"(a8)")"In_check"
    write(154,"(a1)")"1"
    write(154,"(a3)")"1.0"
    write(154,"(a1)")"3"
    write(154,"(a1)")"0"
    write(154,"(a1)")"1"
    write(154,*) mshnd%n

    open(155,FILE="Al_map.out")
    write(155,"(a11)")"$MeshFormat"
    write(155,"(a5)")"2 0 8"
    write(155,"(a9)")"$NodeData"
    write(155,"(a1)")"1"
    write(155,"(a8)")"Al_check"
    write(155,"(a1)")"1"
    write(155,"(a3)")"1.0"
    write(155,"(a1)")"3"
    write(155,"(a1)")"0"
    write(155,"(a1)")"1"
    write(155,*) mshnd%n


    diffuse_length = 10.0e-8 ! 1nm

    write(*,*) "MESH INTERPOLATION map="
    test=1
    do i=1,mshnd%n ! Eg_check.out
        nodetype = mshnd%element(i) ! Get the node number
        x=mshnd%x(i)
        y=mshnd%y(i)
        z=mshnd%z(i)

        if((qw_begin(1)*1e-8-diffuse_length) <=z .and. z<= (qw_end(1)*1e-8+diffuse_length)) then
            test = 1
            call interpolation_qw(Incom_inter,gmapqw,x,y,z,grid_x,grid_y,grid_z,largeW,test,qw_begin,qwnum,Nx,Ny,Nqw)
            write(154,*)  i ,Incom_inter
        else
            Incom_inter=0.0
            write(154,*)  i ,Incom_inter
        end if

       
        if((cap_begin(1)*1e-8-diffuse_length) <=z .and. z<= (cap_end(1)*1e-8)+diffuse_length) then
            test = 1
            call interpolation_cap(Alcom_inter,gmapcap,x,y,z,grid_x,grid_y,grid_z,largeW,test,cap_begin,capnum,Nx,Ny,Ncap)
            write(155,*)  i ,Alcom_inter
        else
            Alcom_inter=0.0
            write(155,*)  i ,Alcom_inter
        end if
    end do


    !do j=mshel%tetraelenumstart,mshel%n
    !   material%EgEl(j)=(material%Eg(mshel%nl(1,j))+material%Eg(mshel%nl(2,j))+material%Eg(mshel%nl(3,j))+material%Eg(mshel%nl(4,j)))/4.0
    !end do


    write(153,*) "$EndNodeData"
    write(154,*) "$EndNodeData"
    write(155,*) "$EndNodeData"

    close(153)
    close(154)
    close(155)
999 format(1000F11.5)

    print*, "You have activated the program to generation composition map by an external function!"
    print*, "You have to run the program with 3D-ddcc-dyna-new.exe. In adition, please remember to modify "
    print*, "libcompositionmapgen.f90  and recompile it! Program stop"
    !stop
    end subroutine

    subroutine interpolation_qw(Incom_inter,gmapqw,x,y,z,grid_x,grid_y,grid_z,largeW,test,qw_begin,qwnum,Nx,Ny,Nqw)
    use meshelement
    use global3d
    implicit none

    integer(i8b) :: test,qwnum,Nx,Ny,Nqw
    real*8 ::grid_x,grid_y,grid_z,Incom_inter,x,y,z
    real*8 ::gmapqw(Nx,Ny,Nqw,qwnum)
    real*8 ::interpolation_x(4) , interpolation_y(2),qw_begin(qwnum)
    real*8, allocatable ::  x_origin_position(:,:),y_origin_position(:,:),z_origin_position(:,:)
    integer::largeW,xp_ordinal,yp_ordinal,zp_ordinal,iii,jjj,kkk,pp

    Incom_inter=0.0
    allocate(x_origin_position(Nx+1,qwnum),y_origin_position(Ny+1,qwnum),z_origin_position(Nqw+1,qwnum))
    do pp=1,qwnum
        do iii = 1, Nx+1
            x_origin_position(iii,pp) = (iii-1) * grid_x
        end do
        do jjj = 1, Ny+1
            y_origin_position(jjj,pp) = (jjj-1) * grid_y
        end do
        do kkk = 1, Nqw+1
            z_origin_position(kkk,pp) = (kkk-1-largeW/2) * grid_z
        end do
    end do
    xp_ordinal  = int((x*1.0e8)/grid_x)
    yp_ordinal  = int((y*1.0e8)/grid_y)
    zp_ordinal  = int((z*1.0e8-qw_begin(test))/grid_z+(largeW/2))
    interpolation_x(1) = gmapqw(xp_ordinal+1,yp_ordinal+1,zp_ordinal+1,test) &
        +(x*1.0e8-x_origin_position(xp_ordinal+1,test))/(x_origin_position(xp_ordinal+2,test)-x_origin_position(xp_ordinal+1,test))  &
        *(gmapqw(xp_ordinal+1+1,yp_ordinal+1,zp_ordinal+1,test)-gmapqw(xp_ordinal+1,yp_ordinal+1,zp_ordinal+1,test))

    interpolation_x(2) = gmapqw(xp_ordinal+1,yp_ordinal+2,zp_ordinal+1,test) &
        +(x*1.0e8-x_origin_position(xp_ordinal+1,test))/(x_origin_position(xp_ordinal+2,test)-x_origin_position(xp_ordinal+1,test))  &
        *(gmapqw(xp_ordinal+1+1,yp_ordinal+2,zp_ordinal+1,test)-gmapqw(xp_ordinal+1,yp_ordinal+2,zp_ordinal+1,test))

    interpolation_x(3) = gmapqw(xp_ordinal+1,yp_ordinal+1,zp_ordinal+2,test) &
        +(x*1.0e8-x_origin_position(xp_ordinal+1,test))/(x_origin_position(xp_ordinal+2,test)-x_origin_position(xp_ordinal+1,test))  &
        *(gmapqw(xp_ordinal+1+1,yp_ordinal+1,zp_ordinal+2,test)-gmapqw(xp_ordinal+1,yp_ordinal+1,zp_ordinal+2,test))

    interpolation_x(4) = gmapqw(xp_ordinal+1,yp_ordinal+2,zp_ordinal+2,test) &
        +(x*1.0e8-x_origin_position(xp_ordinal+1,test))/(x_origin_position(xp_ordinal+2,test)-x_origin_position(xp_ordinal+1,test))  &
        *(gmapqw(xp_ordinal+1+1,yp_ordinal+2,zp_ordinal+2,test)-gmapqw(xp_ordinal+1,yp_ordinal+2,zp_ordinal+2,test))

    interpolation_y(1) = interpolation_x(1)+(y*1.0e8-y_origin_position(yp_ordinal+1,test))/(y_origin_position(yp_ordinal+2,test)-y_origin_position(yp_ordinal+1,test)) &
        *(interpolation_x(2)-interpolation_x(1))

    interpolation_y(2) = interpolation_x(3)+(y*1.0e8-y_origin_position(yp_ordinal+1,test))/(y_origin_position(yp_ordinal+2,test)-y_origin_position(yp_ordinal+1,test)) &
        *(interpolation_x(4)-interpolation_x(3))

    Incom_inter = interpolation_y(1)+( z*1.0e8-qw_begin(test)-z_origin_position(zp_ordinal+1,test))/(z_origin_position(zp_ordinal+2,test)-z_origin_position(zp_ordinal+1,test)) &
        *(interpolation_y(2)-interpolation_y(1))
    return
    end subroutine interpolation_qw




    subroutine interpolation_cap(Alcom_inter,gmapcap,x,y,z,grid_x,grid_y,grid_z,largeW,test,cap_begin,capnum,Nx,Ny,Ncap)
    use meshelement
    use global3d
    implicit none

    integer(i8b) :: test,capnum,Nx,Ny,Ncap
    real*8 ::grid_x,grid_y,grid_z,Alcom_inter,x,y,z
    real*8 ::gmapcap(Nx,Ny,Ncap,capnum)
    real*8 ::interpolation_x(4) , interpolation_y(2),cap_begin(capnum)
    real*8, allocatable ::  x_origin_position(:,:),y_origin_position(:,:),z_origin_position(:,:)
    integer::largeW,xp_ordinal,yp_ordinal,zp_ordinal,iii,jjj,kkk,pp

    Alcom_inter=0.0
    allocate(x_origin_position(Nx+1,capnum),y_origin_position(Ny+1,capnum),z_origin_position(Ncap+1,capnum))
    do pp=1,capnum
        do iii = 1, Nx+1
            x_origin_position(iii,pp) = (iii-1) * grid_x
        end do
        do jjj = 1, Ny+1
            y_origin_position(jjj,pp) = (jjj-1) * grid_y
        end do
        do kkk = 1, Ncap+1
            z_origin_position(kkk,pp) = (kkk-1-largeW/2) * grid_z
        end do
    end do
    xp_ordinal  = int((x*1.0e8)/grid_x)
    yp_ordinal  = int((y*1.0e8)/grid_y)
    zp_ordinal  = int((z*1.0e8-cap_begin(test))/grid_z+(largeW/2))
    interpolation_x(1) = gmapcap(xp_ordinal+1,yp_ordinal+1,zp_ordinal+1,test) &
        +(x*1.0e8-x_origin_position(xp_ordinal+1,test))/(x_origin_position(xp_ordinal+2,test)-x_origin_position(xp_ordinal+1,test))  &
        *(gmapcap(xp_ordinal+1+1,yp_ordinal+1,zp_ordinal+1,test)-gmapcap(xp_ordinal+1,yp_ordinal+1,zp_ordinal+1,test))

    interpolation_x(2) = gmapcap(xp_ordinal+1,yp_ordinal+2,zp_ordinal+1,test) &
        +(x*1.0e8-x_origin_position(xp_ordinal+1,test))/(x_origin_position(xp_ordinal+2,test)-x_origin_position(xp_ordinal+1,test))  &
        *(gmapcap(xp_ordinal+1+1,yp_ordinal+2,zp_ordinal+1,test)-gmapcap(xp_ordinal+1,yp_ordinal+2,zp_ordinal+1,test))

    interpolation_x(3) = gmapcap(xp_ordinal+1,yp_ordinal+1,zp_ordinal+2,test) &
        +(x*1.0e8-x_origin_position(xp_ordinal+1,test))/(x_origin_position(xp_ordinal+2,test)-x_origin_position(xp_ordinal+1,test))  &
        *(gmapcap(xp_ordinal+1+1,yp_ordinal+1,zp_ordinal+2,test)-gmapcap(xp_ordinal+1,yp_ordinal+1,zp_ordinal+2,test))

    interpolation_x(4) = gmapcap(xp_ordinal+1,yp_ordinal+2,zp_ordinal+2,test) &
        +(x*1.0e8-x_origin_position(xp_ordinal+1,test))/(x_origin_position(xp_ordinal+2,test)-x_origin_position(xp_ordinal+1,test))  &
        *(gmapcap(xp_ordinal+1+1,yp_ordinal+2,zp_ordinal+2,test)-gmapcap(xp_ordinal+1,yp_ordinal+2,zp_ordinal+2,test))

    interpolation_y(1) = interpolation_x(1)+(y*1.0e8-y_origin_position(yp_ordinal+1,test))/(y_origin_position(yp_ordinal+2,test)-y_origin_position(yp_ordinal+1,test)) &
        *(interpolation_x(2)-interpolation_x(1))

    interpolation_y(2) = interpolation_x(3)+(y*1.0e8-y_origin_position(yp_ordinal+1,test))/(y_origin_position(yp_ordinal+2,test)-y_origin_position(yp_ordinal+1,test)) &
        *(interpolation_x(4)-interpolation_x(3))

    Alcom_inter = interpolation_y(1)+( z*1.0e8-cap_begin(test)-z_origin_position(zp_ordinal+1,test))/(z_origin_position(zp_ordinal+2,test)-z_origin_position(zp_ordinal+1,test)) &
        *(interpolation_y(2)-interpolation_y(1))
    return
    end subroutine interpolation_cap


    subroutine interpolation_ebl(Alcom_inter,gmapebl,x,y,z,grid_x,grid_y,grid_z,largeW,test,ebl_begin,eblnum,Nx,Ny,Nebl)
    use meshelement
    use global3d
    implicit none

    integer(i8b) :: test,eblnum,Nx,Ny,Nebl
    real*8 ::grid_x,grid_y,grid_z,Alcom_inter,x,y,z
    real*8 ::gmapebl(Nx,Ny,Nebl,eblnum)
    real*8 ::interpolation_x(4) , interpolation_y(2),ebl_begin(eblnum)
    real*8, allocatable ::  x_origin_position(:,:),y_origin_position(:,:),z_origin_position(:,:)
    integer::largeW,xp_ordinal,yp_ordinal,zp_ordinal,iii,jjj,kkk,pp

    Alcom_inter=0.0
    allocate(x_origin_position(Nx+1,eblnum),y_origin_position(Ny+1,eblnum),z_origin_position(Nebl+1,eblnum))
    do pp=1,eblnum
        do iii = 1, Nx+1
            x_origin_position(iii,pp) = (iii-1) * grid_x
        end do
        do jjj = 1, Ny+1
            y_origin_position(jjj,pp) = (jjj-1) * grid_y
        end do
        do kkk = 1, Nebl+1
            z_origin_position(kkk,pp) = (kkk-1-largeW/2) * grid_z
        end do
    end do
    xp_ordinal  = int((x*1.0e8)/grid_x)
    yp_ordinal  = int((y*1.0e8)/grid_y)
    zp_ordinal  = int((z*1.0e8-ebl_begin(test))/grid_z+(largeW/2))
    interpolation_x(1) = gmapebl(xp_ordinal+1,yp_ordinal+1,zp_ordinal+1,test) &
        +(x*1.0e8-x_origin_position(xp_ordinal+1,test))/(x_origin_position(xp_ordinal+2,test)-x_origin_position(xp_ordinal+1,test))  &
        *(gmapebl(xp_ordinal+1+1,yp_ordinal+1,zp_ordinal+1,test)-gmapebl(xp_ordinal+1,yp_ordinal+1,zp_ordinal+1,test))

    interpolation_x(2) = gmapebl(xp_ordinal+1,yp_ordinal+2,zp_ordinal+1,test) &
        +(x*1.0e8-x_origin_position(xp_ordinal+1,test))/(x_origin_position(xp_ordinal+2,test)-x_origin_position(xp_ordinal+1,test))  &
        *(gmapebl(xp_ordinal+1+1,yp_ordinal+2,zp_ordinal+1,test)-gmapebl(xp_ordinal+1,yp_ordinal+2,zp_ordinal+1,test))

    interpolation_x(3) = gmapebl(xp_ordinal+1,yp_ordinal+1,zp_ordinal+2,test) &
        +(x*1.0e8-x_origin_position(xp_ordinal+1,test))/(x_origin_position(xp_ordinal+2,test)-x_origin_position(xp_ordinal+1,test))  &
        *(gmapebl(xp_ordinal+1+1,yp_ordinal+1,zp_ordinal+2,test)-gmapebl(xp_ordinal+1,yp_ordinal+1,zp_ordinal+2,test))

    interpolation_x(4) = gmapebl(xp_ordinal+1,yp_ordinal+2,zp_ordinal+2,test) &
        +(x*1.0e8-x_origin_position(xp_ordinal+1,test))/(x_origin_position(xp_ordinal+2,test)-x_origin_position(xp_ordinal+1,test))  &
        *(gmapebl(xp_ordinal+1+1,yp_ordinal+2,zp_ordinal+2,test)-gmapebl(xp_ordinal+1,yp_ordinal+2,zp_ordinal+2,test))

    interpolation_y(1) = interpolation_x(1)+(y*1.0e8-y_origin_position(yp_ordinal+1,test))/(y_origin_position(yp_ordinal+2,test)-y_origin_position(yp_ordinal+1,test)) &
        *(interpolation_x(2)-interpolation_x(1))

    interpolation_y(2) = interpolation_x(3)+(y*1.0e8-y_origin_position(yp_ordinal+1,test))/(y_origin_position(yp_ordinal+2,test)-y_origin_position(yp_ordinal+1,test)) &
        *(interpolation_x(4)-interpolation_x(3))

    Alcom_inter = interpolation_y(1)+( z*1.0e8-ebl_begin(test)-z_origin_position(zp_ordinal+1,test))/(z_origin_position(zp_ordinal+2,test)-z_origin_position(zp_ordinal+1,test)) &
        *(interpolation_y(2)-interpolation_y(1))
    return
    end subroutine interpolation_ebl
import numpy
from scipy import weavesci
import pylab

################################################################################


class SimpleNEB:
    '''
    A simple 2D system for NEB demonstration
    '''
    
    def __init__( self ):
        '''
        Class initialization.
        
        All the configuration is done here.
        You can change it here or override it from the outside later,
        in that case make sure you call all the stuff that depends on
        the change, e.g. call InterpolateImages() after changing the
        positions of the first and last image.
        '''
        
        # whether to plot the force field
        self.ForceField = False
        
        # image count
        self.ic = 10
        
        # potential plot resolution
        self.resolution = 300
        
        # allocate arrays
        self.x = numpy.zeros( (self.ic, 2) )
        self.v = numpy.zeros( (self.ic, 2) )
        self.t = numpy.zeros( (self.ic, 2) )
        self.E = numpy.zeros( self.ic )
        self.bE = numpy.zeros( self.ic-1 )
        self.k = numpy.zeros( self.ic-1 ) 
        

        # choose the potential and corresponding force
        self.V = self.V_LEPS_weave
        self.F = self.F_LEPS_weave
        # these are the python, non-scipy.weave versions,
        # use them, if you don't have scipy available
        #self.V = self.V_LEPS
        #self.F = self.F_LEPS
        
        # set plot limits
        self.limits = [0.3, 4, 0.3, 4]

        # set the first and the last image
        self.x[0][0] = 0.745
        self.x[0][1] = 3.75
        self.x[self.ic-1][0] = 3.75
        self.x[self.ic-1][1] = 0.75
        
        # for potentials with local minima, call these
        #self.RelaxImage( 0 )
        #self.RelaxImage( self.ic-1 )
        
        # set the intermediate images
        self.InterpolateImages()
        
        # precompute the potential surface matrix
        self.UpdatePotentialMatrix()
        
        # set max of the potential for the potential surface plot
        self.potmax=self.matrix[self.resolution-1][self.resolution-1]
        
        # NEB "time" step
        #self.dt = 1e-1
	self.dt = 0.1
        
        # maximum k and the range of k
        # ks get interpolated for the intermediate images, see article
        self.kmax = 40
        self.dk = 20
        #self.kmax = 1
        #self.dk=0
        
        # relaxation precission
        self.eps = 1e-4
        
        # when to start CI, smaller means "later"
        self.epsCI = 1e-3

        # precompute the force matrices, if requested
        if self.ForceField:
            self.UpdateForceMatrix()
        self.FFscale = 100
        self.FFresolution = self.resolution/4

        
    def V_LEPS_weave( self, x, y ):       
        '''
        potential energy as a function of position
        for the LEPS potential on a line
        scipy.weave version
        '''
        
        support = '''
                 #include <math.h>

                 const double a = 0.05;
                 const double b = 0.3;
                 const double c = 0.05;
                 const double alpha = 1.942;
                 const double r0 = 0.742;
                 const double dAB = 4.746;
                 const double dBC = 4.746;
                 const double dAC = 3.445;


                 double Q( double d, double r ) {
                     return d*( 3*exp(-2*alpha*(r-r0))/2 - exp(-alpha*(r-r0)) )/2;
                 }
               
                 double J( double d, double r ) {
                     return d*( exp(-2*alpha*(r-r0)) - 6*exp(-alpha*(r-r0)) )/4;
                 }
                 
                 '''
        
        code = '''
               double rAB = x;
               double rBC = y;
               double rAC = rAB + rBC;
               
               double JABred = J(dAB, rAB)/(1+a);
               double JBCred = J(dBC, rBC)/(1+b);
               double JACred = J(dAC, rAC)/(1+c);
                              
               return_val = Q(dAB, rAB)/(1+a) +
                            Q(dBC, rBC)/(1+b) +
                            Q(dAC, rAC)/(1+c) -
                            sqrt(
                            JABred*JABred +
                            JBCred*JBCred +
                            JACred*JACred -
                            JABred*JBCred -
                            JBCred*JACred -
                            JABred*JACred
                            );
               '''

        anames = [ 'x', 'y' ]
        
        return scipy.weave.inline( code,
                                   arg_names=anames,
                                   support_code=support )


    def V_LEPS( self, x, y ):       
        '''
        potential energy as a function of position
        for the LEPS potential on a line
        python version
        '''

        a = 0.05
        b = 0.3
        c = 0.05
        alpha = 1.942
        r0 = 0.742
        dAB = 4.746
        dBC = 4.746
        dAC = 3.445

        def Q( d, r ):
            return d*( 3*numpy.exp(-2*alpha*(r-r0))/2 - numpy.exp(-alpha*(r-r0)) )/2
               
        def J( d, r ):
            return d*( numpy.exp(-2*alpha*(r-r0)) - 6*numpy.exp(-alpha*(r-r0)) )/4

        
        rAB = x;
        rBC = y;
        rAC = rAB + rBC;
               
        JABred = J(dAB, rAB)/(1+a)
        JBCred = J(dBC, rBC)/(1+b)
        JACred = J(dAC, rAC)/(1+c)
                              
        return Q(dAB, rAB)/(1+a) + \
               Q(dBC, rBC)/(1+b) + \
               Q(dAC, rAC)/(1+c) - \
               numpy.sqrt( JABred*JABred + \
                           JBCred*JBCred + \
                           JACred*JACred - \
                           JABred*JBCred - \
                           JBCred*JACred - \
                           JABred*JACred )

    
    def F_LEPS_weave( self, x, y ):
        '''
        force as a function of position
        for the LEPS potential on a line
        scipy.weave version
        '''
        
        support = '''
                 #include <math.h>

                 const double a = 0.05;
                 const double b = 0.3;
                 const double c = 0.05;
                 const double alpha = 1.942;
                 const double r0 = 0.742;
                 const double dAB = 4.746;
                 const double dBC = 4.746;
                 const double dAC = 3.445;


                 double Q( double d, double r ) {
                     return d*( 3*exp(-2*alpha*(r-r0))/2 - exp(-alpha*(r-r0)) )/2;
                 }
               
                 double J( double d, double r ) {
                     return d*( exp(-2*alpha*(r-r0)) - 6*exp(-alpha*(r-r0)) )/4;
                 }
                 
                 double dQ( double d, double r ) {
                     return alpha*d*( -3*exp(-2*alpha*(r-r0)) + exp(-alpha*(r-r0)) )/2;
                 }
               
                 double dJ( double d, double r ) {
                     return alpha*d*( -2*exp(-2*alpha*(r-r0)) + 6*exp(-alpha*(r-r0)) )/4;
                 }
                 
                 '''
        
        xcode = '''
                double rAB = x;
                double rBC = y;
                double rAC = rAB + rBC;
               
                double JABred = J(dAB, rAB)/(1+a);
                double JBCred = J(dBC, rBC)/(1+b);
                double JACred = J(dAC, rAC)/(1+c);

                double dJABred = dJ(dAB, rAB)/(1+a);
                double dJBCred = dJ(dBC, rBC)/(1+b);
                double dJACred = dJ(dAC, rAC)/(1+c);
                              
                return_val = dQ(dAB, rAB)/(1+a) +
                             dQ(dAC, rAC)/(1+c) -
                             ( 2*JABred*dJABred +
                               2*JACred*dJACred -
                               dJABred*JBCred -
                               JBCred*dJACred -
                               dJABred*JACred -
                               JABred*dJACred
                             ) /
                             (2*sqrt(
                             JABred*JABred +
                             JBCred*JBCred +
                             JACred*JACred -
                             JABred*JBCred -
                             JBCred*JACred -
                             JABred*JACred
                             ));
                '''

        ycode = '''
                double rAB = x;
                double rBC = y;
                double rAC = rAB + rBC;
               
                double JABred = J(dAB, rAB)/(1+a);
                double JBCred = J(dBC, rBC)/(1+b);
                double JACred = J(dAC, rAC)/(1+c);

                double dJABred = dJ(dAB, rAB)/(1+a);
                double dJBCred = dJ(dBC, rBC)/(1+b);
                double dJACred = dJ(dAC, rAC)/(1+c);
                              
                return_val = dQ(dBC, rBC)/(1+b) +
                             dQ(dAC, rAC)/(1+c) -
                             ( 2*JBCred*dJBCred +
                               2*JACred*dJACred -
                               JABred*dJBCred -
                               dJBCred*JACred -
                               JBCred*dJACred -
                               JABred*dJACred
                             ) /
                             (2*sqrt(
                             JABred*JABred +
                             JBCred*JBCred +
                             JACred*JACred -
                             JABred*JBCred -
                             JBCred*JACred -
                             JABred*JACred
                             ));
                '''

        anames = [ 'x', 'y' ]
        
        Fx = scipy.weave.inline( xcode,
                                 arg_names=anames,
                                 support_code=support )

        Fy = scipy.weave.inline( ycode,
                                 arg_names=anames,
                                 support_code=support )
        
        return ( -Fx, -Fy )



    def F_LEPS( self, x, y ):
        '''
        force as a function of position
        for the LEPS potential on a line
        python version
        '''

        a = 0.05
        b = 0.3
        c = 0.05
        alpha = 1.942
        r0 = 0.742
        dAB = 4.746
        dBC = 4.746
        dAC = 3.445


        def Q( d, r ):
            return d*( 3*numpy.exp(-2*alpha*(r-r0))/2 - numpy.exp(-alpha*(r-r0)) )/2
               
        def J( d, r ):
            return d*( numpy.exp(-2*alpha*(r-r0)) - 6*numpy.exp(-alpha*(r-r0)) )/4
                 
        def dQ( d, r ):
            return alpha*d*( -3*numpy.exp(-2*alpha*(r-r0)) + numpy.exp(-alpha*(r-r0)) )/2;
               
        def dJ( d, r ):
            return alpha*d*( -2*numpy.exp(-2*alpha*(r-r0)) + 6*numpy.exp(-alpha*(r-r0)) )/4;
        
        rAB = x;
        rBC = y;
        rAC = rAB + rBC;
               
        JABred = J(dAB, rAB)/(1+a);
        JBCred = J(dBC, rBC)/(1+b);
        JACred = J(dAC, rAC)/(1+c);

        dJABred = dJ(dAB, rAB)/(1+a);
        dJBCred = dJ(dBC, rBC)/(1+b);
        dJACred = dJ(dAC, rAC)/(1+c);
                              
        Fx = dQ(dAB, rAB)/(1+a) + \
             dQ(dAC, rAC)/(1+c) - \
             ( 2*JABred*dJABred + \
               2*JACred*dJACred - \
               dJABred*JBCred - \
               JBCred*dJACred - \
               dJABred*JACred - \
               JABred*dJACred ) / \
             ( 2 * numpy.sqrt( JABred*JABred + \
                               JBCred*JBCred + \
                               JACred*JACred - \
                               JABred*JBCred - \
                               JBCred*JACred - \
                               JABred*JACred ))


        Fy = dQ(dBC, rBC)/(1+b) + \
             dQ(dAC, rAC)/(1+c) - \
             ( 2*JBCred*dJBCred + \
               2*JACred*dJACred - \
               JABred*dJBCred - \
               dJBCred*JACred - \
               JBCred*dJACred - \
               JABred*dJACred ) / \
             ( 2 * numpy.sqrt( JABred*JABred + \
                               JBCred*JBCred + \
                               JACred*JACred - \
                               JABred*JBCred - \
                               JBCred*JACred - \
                               JABred*JACred ))

        return ( -Fx, -Fy )
    
    
    def RelaxImage( self, imnum ):
        '''
        relax one image to the local potential minimum
        '''
        
        x = x[imnum]
        diff = 1e5
        Eold = 1e5
        v = 0
        
        while( diff > self.dt*self.eps ):
            E = self.V( x[0], x[1] )
            f = self.F( x[0], x[1] )
            v += self.dt*f
            if numpy.dot( v, f ) < 0:
                v = [ 0, 0 ]
            x += self.dt*v
            diff = abs( E - EOld )
            Eold = E

        x[imnum] = x


    def InterpolateImages( self ):
        '''
        prepare the intermediate images        
        linear interpolation between first and last
        '''
        
        dx1 = (self.x[self.ic-1][0] - self.x[0][0]) / (self.ic-1)
        dx2 = (self.x[self.ic-1][1] - self.x[0][1]) / (self.ic-1)

        for i in range( 1, self.ic-1 ):
            self.x[i][0] = self.x[0][0] + i*dx1
            self.x[i][1] = self.x[0][1] + i*dx2


    def UpdateImageEnergies( self ):
        '''
        Update self.E with energies for current positions.
        '''
        
        for i in range( self.ic ):
            self.E[i] = self.V( self.x[i,0], self.x[i,1] )
    

    def UpdateBondEnergies( self ):
        '''
        Update self.bE with energies of the springs.
        '''
        
        for i in range( self.ic-1 ):
            diff = self.x[i]-self.x[i+1]
            distSq = diff[0]**2 + diff[1]**2
            self.bE[i] = self.k[i] * distSq / 2
    
    
    def UpdateTangentsSimple( self ):
        '''
        Fill self.t with normalized tangents to the images.
        Simple -> half the angle between connecting vectors.
        Tangents point to the image with higher number.
        
        note: do not use this, use UpdateTangentsProper instead
        '''

        for i in range( 1, self.ic-1 ):
            
            # vectors from neighbours
            v1 = self.x[i] - self.x[i-1]
            v2 = self.x[i+1] - self.x[i]
            
            # normalize both
            v1 /= numpy.sqrt( pow(v1[0],2) + pow(v1[1],2) )
            v2 /= numpy.sqrt( pow(v2[0],2) + pow(v2[1],2) )
            
            t = v1 + v2
            self.t[i] = t / numpy.sqrt( pow(t[0],2) + pow(t[1],2) )

            
    def UpdateTangentsProper( self ):
        '''
        better tangents, see the article referenced in the intro
        at the beginning of this file
        '''
        
        E = self.E
        
        for i in range( 1, self.ic-1 ):
            
            # vectors from neighbours
            v1 = self.x[i] - self.x[i-1]
            v2 = self.x[i+1] - self.x[i]
            
            if E[i-1] < E[i] < E[i+1]:
                t = v2
            elif E[i-1] > E[i] > E[i+1]:
                t = v1
            else:            # maximum or minimum
                dhi =  abs( E[i+1] - E[i] )
                dlow = abs( E[i-1] - E[i] )
                dEmax = max( dhi, dlow )
                dEmin = min( dhi, dlow )
                if E[i+1] > E[i-1]:
                    t = v2*dEmax + v1*dEmin
                else:
                    t = v1*dEmax + v2*dEmin

            # normalize and assign
            self.t[i] = t / numpy.sqrt( t[0]**2 + t[1]**2 )


    def NEB( self, plot=False ):
        '''
        Perform the NEB relaxation
        '''
        
        f = numpy.zeros( ( self.ic, 2 ) )    # forces
        Fpot = numpy.zeros( 2 )
        x = self.x
        v = self.v
        t = self.t
        ic = self.ic
        E = self.E
        bE = self.bE
        k = self.k  
        CI = False
        diff = 100
        
        EperIm = 0
        i = 0
        indEmax = 0        
               
        Eref = max( self.V( x[0,0], x[0,1] ),
                    self.V( x[ic-1,0], x[ic-1,1] ) )
        
        Emax = 0
        
        while( diff > self.dt*self.eps ):
            
            # update image energies
            self.UpdateImageEnergies()
            self.UpdateBondEnergies()
            EperImOld = EperIm
            EperIm = ( reduce( lambda x,y: x+y, E ) +
                       reduce( lambda x,y: x+y, bE ) ) / ic
            
            # update tangents
            self.UpdateTangentsProper()
            
            Emax = max( E )
            
            # update spring constants (move this to a method)
            for l in range( ic-1 ):
                Ei = max( E[l], E[l+1] )
                if Ei>Eref:
                    k[l] = self.kmax - self.dk*(Emax-Ei)/(Emax-Eref)
                else:
                    k[l] = self.kmax - self.dk
            
            # calc forces
            for l in range( 1, ic-1 ):

                # spring force
                v1 = x[l+1]-x[l]
                v2 = x[l-1]-x[l]
                Fs = numpy.dot( k[l] * v1 + k[l-1] * v2, t[l] ) * t[l]    # with nudging
                #Fs = k[l] * v1 + k[l-1] * v2                               # without nudging
                     
                # potential force perpendicular to energy path
                Fpot[:] = self.F( x[l,0], x[l,1] )
                Fpot -=  t[l] * numpy.dot( t[l], Fpot )
                
                # total force
                f[l] = Fs + Fpot

            # special treatment for the climbing image
            if CI:
                Fpot[:] = self.F( x[indEmax,0], x[indEmax,1] )
                f[indEmax] = Fpot - 20 * t[indEmax] * numpy.dot( t[indEmax], Fpot )
            
            # update velocities
            for l in range( 1, ic-1 ):
                #v[l] += self.dt*f[l]
                v[l] = self.dt*f[l]

                # force and vel in opposite direction?
                if numpy.dot( v[l], f[l] ) < 0:
                    v[l] = [ 0, 0 ]
            
            # move all the images
            x += self.dt*v
            
            diff = abs( EperIm - EperImOld )
            print "step: " + str(i) + ", EperIm=" + str( EperIm ) + ' D(E)=' + str( diff/self.dt )
            
            #if False:           # for switching off climbing image
            if (CI==False) and (diff<self.dt*self.epsCI):    # CI, energy derivative condition
                CI = True
                for j in range(len(self.E)):
                    if self.E[j] > self.E[indEmax]:
                        indEmax = j
                print "climbing image:", indEmax

            if plot:
        	    self.PlotSystem( tangents=False, filename='NEB-%04i.png' % i )

            i += 1

        print 'E(saddle point): ', E[indEmax]

    
    def UpdatePotentialMatrix( self ):
        '''
        Prepare the potential matrix for plotting
        '''
        
        print "updating the potential matrix...",
        
        n = self.resolution

        dx = 1.0*(self.limits[1] - self.limits[0]) / (n-1)
        dy = 1.0*(self.limits[3] - self.limits[2]) / (n-1)
    
        data = numpy.zeros( (n,n) )
        
        for i in range(n):
            for j in range(n):
                data[i][j] = self.V( self.limits[0]+i*dx, self.limits[2]+j*dy )

        # trans because the last index goes along a row (as x coordinate)
        self.matrix = data.transpose()
        
        print "done."

        
    def UpdateForceMatrix( self ):
        '''
        '''
        
        print "updating the force matrix...",
        
        n = self.FFresolution
        Fxdata = numpy.zeros( (n,n) )
        Fydata = numpy.zeros( (n,n) )
        xdata = numpy.zeros( (n,n) )
        ydata = numpy.zeros( (n,n) )
        dx = 1.0*(self.limits[1] - self.limits[0]) / (n-1)
        dy = 1.0*(self.limits[3] - self.limits[2]) / (n-1)

        for i in range(n):
            for j in range(n):        
                ( Fxdata[i][j], Fydata[i][j] ) = self.F( self.limits[0]+i*dx, self.limits[2]+j*dy )
                xdata[i][j] = self.limits[0]+i*dx
                ydata[i][j] = self.limits[0]+j*dy

        # trans because the last index goes along a row (as x coordinate)                
        self.Fxmatrix = Fxdata.transpose()
        self.Fymatrix = Fydata.transpose()
        self.xmatrix = xdata.transpose()
        self.ymatrix = ydata.transpose()
        
        print "done."


    def PlotEnergyProfile( self, distances=False, filename=None ):
        '''
        plot the energy profile
        print image distances - optional

        if filename given, save to file
        '''

        # prepare energy profile data
        pathx = numpy.zeros( self.ic )
        pathx[0] = 0
        for i in range( 1, self.ic ):
            pathx[i] = pathx[i-1] + numpy.sqrt( pow( self.x[i,0]-self.x[i-1,0], 2 ) +
                                                pow( self.x[i,1]-self.x[i-1,1], 2 ) )

        if distances:
            print
            print "image distances:"
            for i in range( self.ic-1 ):
                print pathx[i+1]-pathx[i]

        pylab.figure( 1, dpi=100 )
        #pylab.plot( pathx, self.E, 'k-' )
        pylab.plot( pathx, self.E, 'ko' )

        if filename == None:
            pylab.show()
        else:
            pylab.savefig( filename )
            pylab.close( 1 )
        
        
    def PlotSystem( self, tangents=False, filename=None ):
        '''
        plot the whole 2D system
         * energy path
         * potential
         * force field - optional
         * tangents - optional

        if filename given, save to file
        '''

        # potential plot
        pylab.figure( 1, dpi=100 )
        pylab.subplots_adjust( top=0.95,bottom=0.05,left=0.05,right=0.95 )
        pylab.imshow( self.matrix,
                      origin='lower',
                      vmax=self.potmax,
                      extent=self.limits,
                      cmap=pylab.cm.jet )
        pylab.colorbar()
        
        # force field 
        if self.ForceField:
            pylab.quiver( self.xmatrix,
                          self.ymatrix,
                          self.Fxmatrix,
                          self.Fymatrix,
                          scale=FFscale )
    
        # tangents
        if tangents:
            pylab.quiver( self.x[:,0],
                          self.x[:,1],
                          self.t[:,0],
                          self.t[:,1],
                          scale=20 )
        pylab.plot( self.x[:,0], self.x[:,1], 'k-' )
        pylab.plot( self.x[:,0], self.x[:,1], 'ko' )
        pylab.axis( self.limits )

        if filename == None:
            pylab.show()
        else:
            pylab.savefig( filename )
            pylab.close()


################################################################################


if __name__ == "__main__":
    
    system = SimpleNEB()

    system.NEB( plot=False )
    system.PlotSystem( tangents=False )
    system.PlotEnergyProfile()

    #system.NEB( plot=True )
    #system.PlotEnergyProfile( filename='profile.png' )
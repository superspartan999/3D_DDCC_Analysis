import numpy as np
from qmsolve import Hamiltonian,  SingleParticle, init_visualization, Å, eV


#=========================================================================================================#
#We define the Hamiltonian of a single particle confined in an harmonic oscillator potential. 
#Then, we compute its eigenstates.
#=========================================================================================================#


#interaction potential
def harmonic_oscillator(particle):
 	k = 100 * eV / Å**2
 	return 0.5 * k * particle.x**2

def zero(particle):
    
    return particle.x*0

#define the Hamiltonian
H = Hamiltonian(particles = SingleParticle(), 
				potential = harmonic_oscillator, 
				spatial_ndim = 1, N = 512, extent = 20*Å)

#Diagonalize the Hamiltonian and compute the eigenstates
eigenstates = H.solve(max_states = 500)

print(eigenstates.energies) # the printed energies are expressed in eV

sliders=[]
# Visualize the Eigenstates
visualization = init_visualization(eigenstates)
visualization.slider_plot() #interactive slider

# # (Optional: Visualize a specific eigenstate)




# #interaction potential
# def harmonic_oscillator(particle):

# 	kx = 0.02 
# 	ky = 0.02
# 	return 0 * kx * particle.x**2    +    0 * ky * particle.y**2



# H = Hamiltonian(particles = SingleParticle(), 
# 				potential = harmonic_oscillator, 
# 				spatial_ndim = 2, N = 100, extent = 15e-10)


# eigenstates = H.solve(max_states = 30)

# print(eigenstates.energies)
# visualization = init_visualization(eigenstates)
# #visualization.plot_eigenstate(6)
# # visualization.slider_plot()
# #visualization.animate()
# coeffs = np.zeros([10], np.complex128)
# coeffs[1] = 1.0
# coeffs[2] = 1.0j
# visualization.superpositions(coeffs, dt=0.01, 
# # 							 save_animation=True, frames=60
# 							 )
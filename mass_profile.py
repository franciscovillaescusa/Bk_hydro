from mpi4py import MPI
import numpy as np
import snapshot as sn
import readsnapHDF5 as rs
import HI_library as HIL
import sys,os,glob,h5py,time
import MAS_library as MASL
import HI.HI_image_library as HIIL
import groupcat
import sorting_library as SL

####### MPI DEFINITIONS #######
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()


################################ INPUT ########################################
#run = '/n/hernquistfs3/IllustrisTNG/Runs/L205n2500TNG'
run = '/n/hernquistfs3/IllustrisTNG/Runs/L205n2500TNG_DM'
#run = '/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG'
#run = '/n/hernquistfs3/IllustrisTNG/Runs/L75n910TNG'

snapnum = 99 #17(z=5) 21(z=4) 25(z=3) 33(z=2) 50(z=1) 99(z=0)

Mmin = 1.0e11  #Msun/h
Mmax = 1.0e15  #Msun/h

cell_size = 2.0  #Mpc/h must be larger than virial radii of the considered halos
bins      = 50   #number of bins in the density profile
R1        = 1e-3 #Mpc/h first bin in the profiles goes from 0 to R1
###############################################################################

# find offset_root and snapshot_root                             
snapshot_root = '%s/output/'%run

# read header
snapshot = '%s/output/snapdir_%03d/snap_%03d'%(run,snapnum, snapnum)
f        = h5py.File(snapshot+'.0.hdf5', 'r')
redshift = f['Header'].attrs[u'Redshift']
BoxSize  = f['Header'].attrs[u'BoxSize']/1e3  #Mpc/h
filenum  = f['Header'].attrs[u'NumFilesPerSnapshot']
Omega_m  = f['Header'].attrs[u'Omega0']
Omega_L  = f['Header'].attrs[u'OmegaLambda']
h        = f['Header'].attrs[u'HubbleParam']
Masses   = f['Header'].attrs[u'MassTable']*1e10  #Msun/h
f.close()

if myrank==0:
    print '\nBoxSize         = %.1f Mpc/h'%BoxSize
    print 'Number of files = %d'%filenum
    print 'Omega_m         = %.3f'%Omega_m
    print 'Omega_l         = %.3f'%Omega_L
    print 'redshift        = %.3f'%redshift

# find the output name
fout1 = 'profiles_TNG300-1_DM_%.1e-%.1e_%d_z=%.2f.hdf5'%(Mmin,Mmax,bins,redshift)

# read number of particles in halos and subhalos and number of subhalos
if myrank==0:  print '\nReading halo catalogue...'
halos = groupcat.loadHalos(snapshot_root, snapnum, 
                           fields=['GroupPos','GroupMass',
                                   'Group_R_TopHat200','Group_M_TopHat200'])
halo_pos  = halos['GroupPos']/1e3           #Mpc/h
halo_R    = halos['Group_R_TopHat200']/1e3  #Mpc/h
halo_mass = halos['Group_M_TopHat200']*1e10 #Msun/h
#halo_mass = halos['GroupMass']*1e10        #Msun/h
del halos

# consider only halos in the mass range and with R>0
indexes   = np.where((halo_mass>Mmin) & (halo_mass<Mmax) & (halo_R>0.0))[0]
halo_pos  = halo_pos[indexes]
halo_R    = halo_R[indexes]
halo_mass = halo_mass[indexes]

if myrank==0:
    print 'Found %d halos with masses %.2e < M < %.2e'\
        %(len(indexes), np.min(halo_mass), np.max(halo_mass))
    print 'Radii in the range %.5f < R < %.5f'%(np.min(halo_R), np.max(halo_R))
    print 'Using a cell size of %.3f Mpc/h'%cell_size

if np.max(halo_R)>cell_size:
    raise Exception("cell size should be larger than biggest halo radius!!!")

comm.Barrier() # just to make the above output clear

# sort halo positions and find their ids
data = SL.sort_3D_pos(halo_pos, BoxSize, cell_size, return_indexes=True, 
		return_offset=False)
halo_pos  = data.pos_sorted
halo_R    = halo_R[data.indexes]
halo_mass = halo_mass[data.indexes]
halos     = halo_pos.shape[0]

# find the id = dims2*i + dims*j + k of the cell where halo is
halo_id = SL.indexes_3D_cube(halo_pos, BoxSize, cell_size)

# define the arrays containing the mass and number of particles in each spherical shell
mass_shell_gas     = np.zeros((halos, bins), dtype=np.float64)
mass_shell_CDM     = np.zeros((halos, bins), dtype=np.float64)
mass_shell_stars   = np.zeros((halos, bins), dtype=np.float64)
mass_shell_bh      = np.zeros((halos, bins), dtype=np.float64)
part_in_halo_gas   = np.zeros(halos,         dtype=np.int64)
part_in_halo_CDM   = np.zeros(halos,         dtype=np.int64)
part_in_halo_stars = np.zeros(halos,         dtype=np.int64)
part_in_halo_bh    = np.zeros(halos,         dtype=np.int64)

# do a loop over each subsnapshot
numbers = np.where(np.arange(filenum)%nprocs==myrank)[0]
for i in numbers:

    # find subfile name and read the number of particles in it
    snapshot = snapshot_root + 'snapdir_%03d/snap_%03d.%d'%(snapnum, snapnum, i)
    header   = rs.snapshot_header(snapshot)
    npart    = header.npart 

    ###################### GAS #######################
    pos  = rs.read_block(snapshot, 'POS ', parttype=0, verbose=False)/1e3
    pos  = pos.astype(np.float32)
    mass = rs.read_block(snapshot, 'MASS', parttype=0, verbose=False)*1e10

    # sort the positions of the particles
    data = SL.sort_3D_pos(pos, BoxSize, cell_size, return_indexes=True,
                          return_offset=True)
    pos        = data.pos_sorted
    mass       = mass[data.indexes]
    offset_gas = data.offset    

    HIL.HI_profile(halo_pos, halo_R, halo_id, pos, mass, offset_gas,
                   mass_shell_gas, part_in_halo_gas, BoxSize, R1)
    ##################################################

    ###################### CDM #######################
    pos  = rs.read_block(snapshot, 'POS ', parttype=1, verbose=False)/1e3
    pos  = pos.astype(np.float32)
    mass = rs.read_block(snapshot, 'MASS', parttype=1, verbose=False)*1e10
    mass = mass.astype(np.float32)

    # sort the positions of the particles
    data = SL.sort_3D_pos(pos, BoxSize, cell_size, return_indexes=True, 
                          return_offset=True)
    pos        = data.pos_sorted
    mass       = mass[data.indexes]
    offset_CDM = data.offset    

    HIL.HI_profile(halo_pos, halo_R, halo_id, pos, mass, offset_CDM,
                   mass_shell_CDM, part_in_halo_CDM, BoxSize, R1)
    ##################################################

    ###################### Stars #####################
    pos  = rs.read_block(snapshot, 'POS ', parttype=4, verbose=False)/1e3
    pos  = pos.astype(np.float32)
    mass = rs.read_block(snapshot, 'MASS', parttype=4, verbose=False)*1e10

    # sort the positions of the particles
    data = SL.sort_3D_pos(pos, BoxSize, cell_size, return_indexes=True, 
                          return_offset=True)
    pos          = data.pos_sorted
    mass         = mass[data.indexes]
    offset_stars = data.offset    

    HIL.HI_profile(halo_pos, halo_R, halo_id, pos, mass, offset_stars,
                   mass_shell_stars, part_in_halo_stars, BoxSize, R1)
    ##################################################

    ################## Black-holes ###################
    pos  = rs.read_block(snapshot, 'POS ', parttype=5, verbose=False)/1e3
    pos  = pos.astype(np.float32)
    mass = rs.read_block(snapshot, 'MASS', parttype=5, verbose=False)*1e10

    # sort the positions of the particles
    data = SL.sort_3D_pos(pos, BoxSize, cell_size, return_indexes=True, 
                          return_offset=True)
    pos       = data.pos_sorted
    mass      = mass[data.indexes]
    offset_bh = data.offset    

    HIL.HI_profile(halo_pos, halo_R, halo_id, pos, mass, offset_bh,
                   mass_shell_bh, part_in_halo_bh, BoxSize, R1)
    ##################################################

    print '\nDone with subfile %03d : %d'%(i,myrank)
    print 'Total mass in gas   so far = %.8e'%(np.sum(mass_shell_gas))
    print 'Total mass in CDM   so far = %.8e'%(np.sum(mass_shell_CDM))
    print 'Total mass in stars so far = %.8e'%(np.sum(mass_shell_stars))
    print 'Total mass in bh    so far = %.8e'%(np.sum(mass_shell_bh))


# sum the results of each indivual core
mass_shell_gas_total   = np.zeros((halos, bins), dtype=np.float64)
mass_shell_CDM_total   = np.zeros((halos, bins), dtype=np.float64)
mass_shell_stars_total = np.zeros((halos, bins), dtype=np.float64)
mass_shell_bh_total    = np.zeros((halos, bins), dtype=np.float64)
comm.Reduce([mass_shell_gas, MPI.DOUBLE],   [mass_shell_gas_total, MPI.DOUBLE], 
            op=MPI.SUM, root=0)
comm.Reduce([mass_shell_CDM, MPI.DOUBLE],   [mass_shell_CDM_total, MPI.DOUBLE], 
            op=MPI.SUM, root=0)
comm.Reduce([mass_shell_stars, MPI.DOUBLE], [mass_shell_stars_total, MPI.DOUBLE], 
            op=MPI.SUM, root=0)
comm.Reduce([mass_shell_bh, MPI.DOUBLE],    [mass_shell_bh_total, MPI.DOUBLE], 
            op=MPI.SUM, root=0)

part_in_halo_gas_total   = np.zeros(halos, dtype=np.int64)
part_in_halo_CDM_total   = np.zeros(halos, dtype=np.int64)
part_in_halo_stars_total = np.zeros(halos, dtype=np.int64)
part_in_halo_bh_total    = np.zeros(halos, dtype=np.int64)
comm.Reduce([part_in_halo_gas, MPI.LONG],   [part_in_halo_gas_total, MPI.LONG], 
            op=MPI.SUM, root=0)
comm.Reduce([part_in_halo_CDM, MPI.LONG],   [part_in_halo_CDM_total, MPI.LONG], 
            op=MPI.SUM, root=0)
comm.Reduce([part_in_halo_stars, MPI.LONG], [part_in_halo_stars_total, MPI.LONG], 
            op=MPI.SUM, root=0)
comm.Reduce([part_in_halo_bh, MPI.LONG],    [part_in_halo_bh_total, MPI.LONG], 
            op=MPI.SUM, root=0)

if myrank==0:

    r          = np.zeros((halos,bins), dtype=np.float64)
    rho_gas    = np.zeros((halos,bins), dtype=np.float64)
    rho_CDM    = np.zeros((halos,bins), dtype=np.float64)
    rho_stars  = np.zeros((halos,bins), dtype=np.float64)
    rho_bh     = np.zeros((halos,bins), dtype=np.float64)
    rho_matter = np.zeros((halos,bins), dtype=np.float64)

    # do a loop over all halos
    for i in xrange(halos):

        # find r-bins and shell volume
        r_bins = np.empty(bins+1, dtype=np.float64)
        r_bins[0] = 1e-15
        r_bins[1:] = np.logspace(np.log10(R1), np.log10(halo_R[i]), bins)
        r[i] = 10**(0.5*(np.log10(r_bins[1:]) + np.log10(r_bins[:-1])))
        V = 4.0*np.pi/3.0*(r_bins[1:]**3 - r_bins[:-1]**3)

        # compute profiles
        rho_gas[i]    = mass_shell_gas_total[i]*1.0/V
        rho_CDM[i]    = mass_shell_CDM_total[i]*1.0/V
        rho_stars[i]  = mass_shell_stars_total[i]*1.0/V
        rho_bh[i]     = mass_shell_bh_total[i]*1.0/V
        rho_matter[i] = (mass_shell_gas_total[i]   + mass_shell_CDM_total[i] +\
                         mass_shell_stars_total[i] + mass_shell_bh_total[i])*1.0/V

    f = h5py.File(fout1, 'w')
    f.create_dataset('r',                 data=r)
    f.create_dataset('rho_gas',           data=rho_gas)
    f.create_dataset('rho_CDM',           data=rho_CDM)
    f.create_dataset('rho_stars',         data=rho_stars)
    f.create_dataset('rho_bh',            data=rho_bh)
    f.create_dataset('rho_matter',        data=rho_matter)
    f.create_dataset('mass_shell_gas',    data=mass_shell_gas_total)
    f.create_dataset('mass_shell_CDM',    data=mass_shell_CDM_total)
    f.create_dataset('mass_shell_stars',  data=mass_shell_stars_total)
    f.create_dataset('mass_shell_bh',     data=mass_shell_bh_total)
    f.create_dataset('Mass',              data=halo_mass)
    f.create_dataset('Particles_gas',     data=part_in_halo_gas_total)
    f.create_dataset('Particles_CDM',     data=part_in_halo_CDM_total)
    f.create_dataset('Particles_stars',   data=part_in_halo_stars_total)
    f.create_dataset('Particles_bh',      data=part_in_halo_bh_total)
    f.close()

"""
    # read HI profiles file
    #f      = h5py.File(fin, 'r')
    #rho_HI = f['rho_HI'][:]
    #Mass   = f['Mass'][:]
    #r      = f['r'][:]
    #f.close()

    Rv_median = np.median(halo_R)
    r_mean_profile = np.logspace(-5, np.log10(Rv_median), bins)

    print 'Median Rv = %.4f Mpc/h'%Rv_median

    f = open(fout, 'w')
    # do a loop over the different radii of the mean profile
    for i in xrange(bins):
	radius = r_mean_profile[i]

        # select the halos with Rv>radius
	indexes      = np.where(halo_R>=radius)[0]
	rho_HI_stack = rho_HI[indexes]
	r_stack      = r[indexes]
	print '%6d halos with Rv>%.5f Mpc/h'%(r_stack.shape[0],radius)

        # interpolate to find rho_HI(r) at radius from contributing halos
	HI_prof = np.zeros(r_stack.shape[0], dtype=np.float64)
	for j in xrange(rho_HI_stack.shape[0]):
		HI_prof[j] = np.interp(radius, r_stack[j], rho_HI_stack[j])

	f.write(str(radius)+' '+str(np.mean(HI_prof))+' '+\
                    str(np.std(HI_prof)/np.sqrt(r_stack.shape[0]))+'\n')
    f.close()
"""

















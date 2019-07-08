import numpy as np
import MAS_library as MASL
import sys,os,h5py,time
import units_library as UL
import HI_library as HIL

U = UL.units();  rho_crit = U.rho_crit #h^2 Msun/Mpc^3
################################ INPUT ########################################
snapnums = np.array([85, 68, 60])

runs  = ['/simons/scratch/sgenel/Illustris_IllustrisTNG_public_data_release/L75n1820FP']
fouts = ['Illustris100-1_matter']

dims = 2048
MAS  = 'CIC'
##############################################################################

for fout,run in zip(fouts,runs):

    # find whether it is an N-body or an hydro sims
    if run[-2:]=='DM':  Nbody = True
    else:               Nbody = False

    # do a loop over the different redshifts
    for snapnum in snapnums:

        # define the array hosting delta_HI and delta_m
        delta_m  = np.zeros((dims,dims,dims), dtype=np.float32)

        # read header
        snapshot = '%s/output/snapdir_%03d/snap_%03d'%(run,snapnum, snapnum)
        f = h5py.File(snapshot+'.0.hdf5', 'r')
        redshift = f['Header'].attrs[u'Redshift']
        BoxSize  = f['Header'].attrs[u'BoxSize']/1e3  #Mpc/h
        filenum  = f['Header'].attrs[u'NumFilesPerSnapshot']
        Omega_m  = f['Header'].attrs[u'Omega0']
        Omega_L  = f['Header'].attrs[u'OmegaLambda']
        h        = f['Header'].attrs[u'HubbleParam']
        Masses   = f['Header'].attrs[u'MassTable']*1e10  #Msun/h
        f.close()

        print 'Working with %s at redshift %.0f'%(run,redshift)
        f_out = '%s_z=%.1f.hdf5'%(fout,round(redshift))
        print f_out

        # if file exists move on
        if os.path.exists(f_out):  continue

        # do a loop over all subfiles in a given snapshot
        M_total, start = 0.0, time.time()
        for i in xrange(filenum):

            snapshot = '%s/output/snapdir_%03d/snap_%03d.%d.hdf5'\
                       %(run,snapnum,snapnum,i)
            f = h5py.File(snapshot, 'r')

            ### CDM ###
            pos  = (f['PartType1/Coordinates'][:]/1e3).astype(np.float32)        
            mass = np.ones(pos.shape[0], dtype=np.float32)*Masses[1] #Msun/h
            MASL.MA(pos, delta_m, BoxSize, MAS, mass)  #CDM
            M_total += np.sum(mass, dtype=np.float64)

            if not(Nbody):

                ### GAS ###
                pos  = (f['PartType0/Coordinates'][:]/1e3).astype(np.float32)
                mass = f['PartType0/Masses'][:]*1e10  #Msun/h
                MASL.MA(pos, delta_m, BoxSize, MAS, mass)  #gas
                M_total += np.sum(mass, dtype=np.float64)

                ### Stars ###
                pos  = (f['PartType4/Coordinates'][:]/1e3).astype(np.float32)        
                mass = f['PartType4/Masses'][:]*1e10  #Msun/h
                MASL.MA(pos, delta_m, BoxSize, MAS, mass)  #stars
                M_total += np.sum(mass, dtype=np.float64)

                ### Black-holes ###
                pos  = (f['PartType5/Coordinates'][:]/1e3).astype(np.float32)        
                mass = f['PartType5/Masses'][:]*1e10  #Msun/h
                MASL.MA(pos, delta_m, BoxSize, MAS, mass)  #black-holes
                M_total += np.sum(mass, dtype=np.float64)

            f.close()

            print '%03d -----> Omega_m = %.4f  : %6.0f s'\
                %(i, M_total/(BoxSize**3*rho_crit), time.time()-start)

        f = h5py.File(f_out,'w')
        f.create_dataset('delta_m',  data=delta_m)
        f.close()

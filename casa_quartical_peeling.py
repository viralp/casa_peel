import os,glob
import numpy as np


masking=True
delay_selfcal=True
ddcal=True

vis='combined.ms'  
#imaging parameters
imsize= [6075, 6075] 
cell= '1.5arcsec'
niter=100000
parallel=False
imagename='relic_target_DI_1'
region_file='bright_src_region.crtf'
field=''
spw=''
mask=''
scales=''

def addModelData(msname, colname):
    tb.open(vis,nomodify=False)
    cnames = tb.colnames()
    # Check if the source column exists
    try:
        cdesc = tb.getcoldesc("DATA")  # Copy column description
    except:
        raise ValueError("Column DATA does not exist")
    hasTiled = False
    dminfo = None
    # Fix the incorrect loop
    try:
        dminfo = tb.getdminfo("DATA")
        if "TYPE" in dminfo and dminfo["TYPE"][:5] == "Tiled":
            hasTiled = True
    except:
        hasTiled = False
    # If not tiled, define default settings
    if not hasTiled:
        dminfo = {"TYPE": "TiledShapeStMan", "SPEC": {"DEFAULTTILESHAPE": [4, 32, 128]}}
    if colname in cnames:
        print(f"Column {colname} not added; it already exists.")
    else:
        dminfo["NAME"] = colname  # Set the new column name
        cdesc["comment"] = "The model data column"
        # Add the column
        tb.addcols({colname: cdesc}, dminfo)
        print(f"Column {colname} added successfully!")

        # Initialize the column with zeros (to prevent 'no array in row 0' error)
        import numpy as np
        nrows = tb.nrows()
        shape = tb.getcell("DATA", 0).shape  # Get data shape
        zero_data = np.zeros(shape, dtype=np.complex64)  # Ensure dtype matches
        for i in range(nrows):
            tb.putcell(colname, i, zero_data)
    tb.flush()
    tb.close()

def remove_column_if_exists(msname, colname):
    tb.open(msname, nomodify=False)
    cnames = tb.colnames()
    if colname in cnames:
        print(f"Removing column: {colname}")
        tb.removecols(colname)
    else:
        print(f"Column {colname} does not exist, skipping removal.")
    tb.close()

def cleaning (vis=vis, imagename=imagename,field=field,spw=spw,
       gridder='wproject',wprojplanes=-1, pblimit=-0.01, imsize=imsize, cell=cell, specmode='mfs',
       deconvolver='mtmfs', nterms=2, scales=scales, smallscalebias=0.9,datacolumn='corrected',
       interactive=False, niter=niter,  weighting='briggs',robust=0,mask=mask,
       stokes='I', threshold='1e-06', savemodel='modelcolumn',parallel=parallel):
       
       tclean(vis=vis, imagename=imagename,field=field,spw=spw,
       gridder=gridder,wprojplanes=wprojplanes, pblimit=pblimit, imsize=imsize, cell=cell, specmode=specmode,
       deconvolver=deconvolver, nterms=nterms, scales=scales, smallscalebias=smallscalebias,datacolumn=datacolumn,
       interactive=False, niter=niter,  weighting=weighting,robust=robust,mask=mask,
       stokes=stokes, threshold=threshold, savemodel=savemodel,parallel=parallel)
       exportfits(imagename=imagename+'.image.tt0',fitsimage=imagename+'.image.tt0.fits',overwrite=True)
       

tb.open(vis, nomodify=False)
cnames = tb.colnames()
if 'CORRECTED_DATA_BKP' in cnames:
  data=tb.getcol("CORRECTED_DATA_BKP")   
  tb.putcol("CORRECTED_DATA", data)
  tb.flush()
  tb.close()
else:    
  addModelData(vis,'CORRECTED_DATA_BKP')
  tb.open(vis,nomodify=False)
  data=tb.getcol("CORRECTED_DATA")
  tb.putcol("CORRECTED_DATA_BKP", data)
  tb.flush()
  tb.close()
    
#remove_column_if_exists(vis, "MODEL_DATA")

if mask==True:
 
 cleaning(imagename=imagename,datacolumn='corrected')
 os.system("breizorro --restored-image "+imagename+".image.tt0.fits --threshold 5 --outfile "+imagename+".sigma.mask.fits")
 importfits(fitsimage=imagename+".mask.fits",imagename=imagename+".sigma.mask")
 mask=imagename+".sigma.mask"

if delay_selfcal==True: 
   #remove_column_if_exists(vis, "MODEL_DATA")
   if os.path.exists(imagename+'.C0'+'.mask'):
        os.system("rm -rf "+str(imagename+'.C0'+'.mask'))
   cleaning(imagename=imagename+'.C0',datacolumn='corrected',mask=mask)
   os.system("goquartical input_ms.path="+str(vis)+" input_ms.data_column=CORRECTED_DATA input_ms.select_fields=["+str(field)+"] input_ms.time_chunk='0' input_ms.freq_chunk='0' input_model.recipe=MODEL_DATA solver.terms='[G,K]' solver.iter_recipe='[50,50,50,50,50,50]' solver.propagate_flags=False solver.robust=False solver.threads=1 solver.convergence_fraction=0.99 solver.convergence_criteria=1e-06 output.log_directory=output/KG.outputs.qc/log output.gain_directory=output/KG.output.gain.qc output.overwrite=1 output.products=[corrected_data,corrected_residual] output.columns=[CORRECTED_DATA,CORRECTED_RESIDUAL] output.flags=False dask.threads=6 dask.workers=6 dask.scheduler=threads G.type=phase G.time_interval='10s' G.freq_interval='0' G.initial_estimate=False G.solve_per=antenna G.interp_mode=reim G.interp_method=2dlinear mad_flags.enable=False mad_flags.threshold_bl=6 mad_flags.threshold_global=8 mad_flags.max_deviation=1000 K.time_interval='1s' K.freq_interval='0' K.type=delay_and_offset K.initial_estimate=False K.interp_mode=reim K.interp_method=2dlinear")
   if os.path.exists(imagename+'.GK_SC'+'.mask'):
        os.system("rm -rf "+str(imagename+'.GK_SC'+'.mask'))
   cleaning(imagename=imagename+'.GK_SC',datacolumn='corrected',mask=mask,savemodel='modelcolumn')
   imagename=imagename+'.GK_SC'
 
if ddcal==True:

 remove_column_if_exists(vis, "ALL_SKY_MODEL")
 addModelData(vis,'ALL_SKY_MODEL')
 tb.open(vis,nomodify=False)
 data=tb.getcol("MODEL_DATA")
 tb.putcol("ALL_SKY_MODEL", data)
 tb.flush()
 tb.close()
 model_images=glob.glob(imagename+'*')
 model_images = [f for f in model_images if not f.endswith('.fits')]
 lines = open(region_file).readlines()
 models = []  # List to store model names
 for i in range(1,len(lines)):
  models.append(f"BRIGHT_Source_MODEL{i}")
  for j in range(len(model_images)):
    inpimage=model_images[j]
    label=model_images[j].replace(imagename,'')
    inpimage=model_images[j].replace(label,'')
    outputimg=inpimage+'_new_model_bright_src'+label
    output_mask=imagename+'_mask_bright_sources.im'
    makemask(mode='copy', inpimage=imagename+'.model.tt0',
         inpmask=region_file, output=output_mask,overwrite=True)
    if os.path.exists(outputimg):
        os.system("rm -rf "+str(outputimg))     
    immath(imagename=[model_images[j], output_mask],expr='IM0*IM1',outfile=outputimg)
  tclean(vis=vis, imagename=imagename+'_new_model_bright_src',field=field,spw=spw,
       gridder='wproject',wprojplanes=-1, pblimit=-0.01, imsize=imsize, cell=cell, specmode='mfs',
       deconvolver='mtmfs', nterms=2, scales=[0], smallscalebias=0.9,datacolumn='corrected',
       interactive=False, niter=0,  weighting='briggs',robust=0,restart=True,
       stokes='I', threshold='1e-06', savemodel='modelcolumn',calcres=False,restoration=False,
       calcpsf=False, parallel=parallel) 
  remove_column_if_exists(vis, f"BRIGHT_Source_MODEL{i}")
  addModelData(vis,f'BRIGHT_Source_MODEL{i}')
  tb.open(vis,nomodify=False)
  data=tb.getcol("MODEL_DATA")  # Get contents of MODEL_DATA column.
  tb.putcol(f"BRIGHT_Source_MODEL{i}", data)  # Put contents of MODEL_DATA into BRIGHT_Source_MODEL.
  tb.flush()
  tb.close()

 if models:  # Ensure there's at least one line after skipping the first
    model_string = f"ALL_SKY_MODEL~{'~'.join(models)}:{':'.join(models)}"

 os.system("goquartical input_ms.path="+str(vis)+" input_ms.data_column=CORRECTED_DATA input_ms.select_fields=["+str(field)+"] input_ms.time_chunk='0' input_ms.select_ddids=["+str(spw)+"] input_ms.freq_chunk='0' input_model.recipe="+str(model_string)+" solver.terms='[G,K,dE]' solver.iter_recipe='[50,50,50,50,50,50,50,50,50,50,50,50]' solver.robust=False solver.propagate_flags=False solver.threads=1 solver.convergence_fraction=0.99 solver.convergence_criteria=1e-06 output.log_directory=output/KG.outputs.qc/log output.gain_directory=output/KG.output.gain.qc output.log_to_terminal=True output.overwrite=True output.products=[corrected_residual] output.columns=[CORRECTED_DATA] output.flags=False output.subtract_directions=[1] dask.threads=6 dask.workers=6 dask.scheduler=threads G.type=diag_complex G.solve_per=antenna G.time_interval='10' G.freq_interval='0' G.initial_estimate=False G.interp_mode=reim G.interp_method=2dlinear G.respect_scan_boundaries=True dE.direction_dependent=True dE.type=complex dE.time_interval='100' dE.freq_interval='0' mad_flags.enable=False mad_flags.threshold_bl=6 mad_flags.threshold_global=8 mad_flags.max_deviation=1000 K.time_interval='1s' K.freq_interval='0' K.type=delay_and_offset K.initial_estimate=False K.interp_mode=reim K.interp_method=2dlinear")
 cleaning(imagename=imagename+'.DD',datacolumn='corrected',mask=mask)






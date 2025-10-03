import os
import datetime
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from astropy.io import fits
import argparse

# Multiprocessing-----------
import multiprocessing as mp
mp.set_start_method('fork')
# --------------------------


# import maven_iuvs as iuvs
# from maven_iuvs.user_paths import idl_pipeline_dir
from maven_iuvs.download import get_default_data_directory
from maven_iuvs.echelle import get_dir_metadata, make_dark_index, \
    find_files_with_geometry, downselect_data, convert_l1a_to_l1c, \
    open_IDL_and_compile_writeout_script, get_dark_from_keyfile
from maven_iuvs.miscellaneous import orbit_folder#, iuvs_orbno_from_fname

DO_WRITEOUT = True # If True, IDL will run.
make_plots = False
save_arrays = False
fitter = "scipy" # "dynesty"
binning = None# "nonlinear"
report_append_text = ""
overwrite = False
# print("WARNING! Redoing nonlinear files. Go back to v15_pipeline and set "
#       "binning=None and overwrite=False if this is not your intention!!")

# ARG PARSE 
# =============================================================================
parser = argparse.ArgumentParser(description='Orbits to process to v15')
parser.add_argument('start_orb', type=int, 
                    help='Start orbit (multiple of 100)')
parser.add_argument('end_orb', type=int, 
                    help='End orbit (multiple of 100) -- will not be included')

args = parser.parse_args()

print(f"Will work on orbits {args.start_orb}--{args.end_orb}")

# STARTING VERSION AND FOLDER DECLARATION
# =============================================================================
v = "v14"
which_l1a = {"v13": "l1a", "v14": "l1a_full_mission_reprocess"}
DO_FMR = True
DO_DISK = False
DO_LIMB = False
DO_PERI = False
DO_CLEANUP_TEST = False

IUVS_FOLD = "/home/emc/Insync/OneDrive-CU/Research/IUVS/"
# IDL_FOLD = IUVS_FOLD + "IDL_pipeline/"
# L1c base 
IUVS_DATA_DIR = "/media/emc/ExtremePro/IUVS/IUVS_Data/"
L1C_DIR = IUVS_DATA_DIR + "l1c_ech_data/FMR_v15/Round3_scipy_parallel/"
            # "l1c_ech_data/disk_survey_ls200-300/v15/" # for disk
          # "l1c_ech_data/Limb_v15/" # for writing new files of record
          # "l1c_ech_data/test_old_cleanup/v15/" # for testing outlier rejection effects
          # "l1c_ech_data/susfile_fits/v15/" #  for limb

keyname = "MASTER_LIGHT_DARK_KEY_v14.csv"
          #input("Please type name of light/dark key to use with .csv: ")
PF = "/home/emc/GITREPOS/maven_iuvs/maven_iuvs/ancillary/" + keyname

# WORKER FUNCTION FOR PARALLELIZATION =========================================
def process_observation(obs_md, subfoldpath, ldkey):
    """
    Load the input files and fit the frames in the observation. Return a status
    """
    # Search it in the CSV
    lfold, ln, dfold, dn = get_dark_from_keyfile(obs_md["name"], ldkey)

    # Light missing
    if ln == "Light missing":
        print("Light not in the key")
        return "Light not in the key"
        

    if dn == "No valid dark found":
        print("No valid dark found")
        return "No valid dark found"

    # Proceed with processing
    try:
        # Open the fits
        lf = fits.open(lfold + ln)
        df = fits.open(dfold + dn)
        
        # Update filename
        l1c_fn = ln[:-8].replace("l1a", "l1c")
        l1c_fn = l1c_fn.replace("v14", "v15")

        # And make a folder for this file for all the line fit plots
        if make_plots:
            PLACE_FOR_FIGS = subfoldpath + f"{l1c_fn}" + "/plot_fits/"
            if not os.path.exists(PLACE_FOR_FIGS):
                os.makedirs(PLACE_FOR_FIGS)
            pk = {"fig_savepath": PLACE_FOR_FIGS}
        else:
            pk = {}

        if save_arrays:
            PLACE_FOR_ARRAYS = subfoldpath + f"{l1c_fn}" + "/plotted_arrays/"
            if not os.path.exists(PLACE_FOR_ARRAYS):
                os.makedirs(PLACE_FOR_ARRAYS)
        else:
            PLACE_FOR_ARRAYS = ""

        # Call the conversion 
        status = convert_l1a_to_l1c(lf, df, lfold+ln, dfold+dn,
                                    subfoldpath, 
                                    fitter=fitter,
                                    overwrite=overwrite,
                                    process_timestamp=process_timestamp,
                                    save_arrays=save_arrays,
                                    place_for_arrays=PLACE_FOR_ARRAYS,
                                    calibration="new", 
                                    make_plots=make_plots,
                                    clean_data_kwargs=clean_kwargs,                       
                                    plot_kwargs=pk,
                                    run_writeout=DO_WRITEOUT,
                                    idl_process_kwargs=idl_process_kwargs,
                                    )

        tf = datetime.datetime.now()

        print(f"Finished with file {obsfile['name']} in {tf-ti} seconds. ")
        print("continuing...")
        print()
        return status 

    except Exception as excep:
        return f"Caught an error: {excep} ...moving on..."


# LOAD LIGHT/DARK PAIR CSV
# =============================================================================
print("Loading light/dark pair CSV")
ld_pairs = pd.read_csv(PF, delimiter=",", header=0)
if ld_pairs.empty:
    raise KeyError("dataframe is empty for some reason")

# LOAD INDICES
# =============================================================================
ech_l1a_idx = get_dir_metadata(get_default_data_directory(which_l1a[v]),
                               geospatial=True)
dark_idx = make_dark_index(ech_l1a_idx)

# Find geometry files
lights_with_geom = find_files_with_geometry(ech_l1a_idx)

# SELECT DATA
# =============================================================================
if DO_FMR: 
    clean_kwargs = {}
    metadata_lists = []
    orbit_folders_to_run =  list(range(args.start_orb, args.end_orb, 100))

    for so in orbit_folders_to_run:
        metadata_lists.append(downselect_data(lights_with_geom,
                                              light_dark="light",
                                              binning=binning,
                                              orbit=[so, so+99]
                                             )
                             )
        # Create the folder so we can open IDL 
        if not os.path.isdir(L1C_DIR + f'orbit{so:05}'):
            makeme = L1C_DIR + f'orbit{so:05}/'
            os.mkdir(makeme)

elif DO_LIMB:
    clean_kwargs = {}
    limbdata_temp = downselect_data(lights_with_geom, light_dark="light", 
                                    segment="limb", 
                                    orbit=[7700, 8000])
    all_metadata = [l for l in limbdata_temp if 'bintbl' not in l['binning']] 
    # all_metadata = np.load(IUVS_FOLD + "notebooks/susfiles_v15.npy",
    #                        allow_pickle=True).item()['ss']
elif DO_DISK:
    clean_kwargs = {}
    # select randomly throughout mission, but using different segments:
    # diskdata = downselect_data(ech_l1a_idx, light_dark="light",
    #                            segment="disk", ls=[200, 300])
    # print(len(diskdata))
    # trimit = input("Trim down the disk data? (y/n)")
    # if trimit=="y":
    #     stepsz = int(input("Enter step size: "))
    #     all_metadata = diskdata[::stepsz]
    # all_metadata = [l for l in all_metadata if 'bintbl' not in l['binning']]
    all_metadata = np.load(IUVS_FOLD + "notebooks/crossmission_diskdata.npy",
                           allow_pickle=True)
elif DO_PERI:
    clean_kwargs = {}
    peridata = downselect_data(ech_l1a_idx, light_dark="light", 
                               segment="periapse", ls=[200, 300])
    print(len(peridata))
    trimit = input("Trim down the peri data? (y/n)")
    if trimit=="y":
        stepsz = int(input("Enter step size: "))
        all_metadata = peridata[::stepsz]
    all_metadata = [l for l in all_metadata if 'bintbl' not in l['binning']]
elif DO_CLEANUP_TEST:
    clean_kwargs = {"clean_method": "old"}
    all_metadata = downselect_data(ech_l1a_idx, light_dark="light",
                               segment="outlimb", 
                               date=datetime.datetime(2020, 9, 26, 20, 45, 28),
                               orbit=12430
                               )

print(f"Total files to process: {sum([len(m) for m in metadata_lists])}")

# IDL 
# =============================================================================
eln = "IDLerrors.txt" 
if DO_WRITEOUT: 
    print("Opening IDL and loading the MAVEN environment")

    # Open a file for output
    # outputfile = open(L1C_DIR + "IDLoutput.txt", "w")
    # errorfile = open(L1C_DIR + "IDLerrors.txt", "w")

    # Open IDL
    # idl_pipeline_folder = idl_pipeline_dir
    # os.chdir(idl_pipeline_dir)
    # idlproc = subprocess.Popen("idl", stdin=subprocess.PIPE,
    #                            stdout=outputfile, stderr=errorfile,
    #                            text=True, bufsize=1) 
    # bufsize=1 forces information to go through pipe to stdin, stdout instead 
    # of getting stuck
    # may need a time.sleep(60*5) here if loading a huge number of kernels

    # print("IDL is now open")

    # Compile our script
    # idlproc.stdin.write(".com write_l1c_file_from_python.pro\n")
    # idlproc.stdin.flush()
    # print("Compiled the script, hopefully anyway")
    # time.sleep(3)
    idlproc, stderr_queue, stderr_thread = open_IDL_and_compile_writeout_script(L1C_DIR, errlogname=eln)

    idl_process_kwargs={"open_idl": False, "proc": idlproc, 
                        "stderr_queue": stderr_queue, 
                        "stderr_thread": stderr_thread}
else:
    idl_process_kwargs={}
    print("File writeout not requested, IDL will not be opened")

# SET UP LOOP 
# =====================================================================================================================================

# get date time here because it's hard to do in IDL
process_timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')


    # with multiprocessing.Pool() as pool:
    #     # Use pool.map() to apply worker_function to each item in data_items
    #     # pool.map distributes the items among the worker processes
    #     results = pool.map(worker_function, data_items)

for i, startorb in enumerate(orbit_folders_to_run):
    this_md = metadata_lists[i]

    # Log problems
    successful_obs = [] #  Now that we track  IDL output we can keep track here
    problem_obs = []
    exception_log = []
    not_in_dark_key = []
    no_dark = [] 

    # Get the orbit subfolder
    orbfold = orbit_folder(startorb)
    orbit_subfolder_path = f"{L1C_DIR}{orbfold}/"

    # Loop over the observations
    for obsfile in tqdm(this_md):
        print(f"Processing {obsfile['name']}")
        ti = datetime.datetime.now()
        
        result = process_observation(obsfile, orbit_subfolder_path, ld_pairs)
        
        # add to lists...
        if result=="OK":
            successful_obs.append(obsfile)
        elif result=="Light not in the key":
            not_in_dark_key.append(obsfile)
        elif result=="No valid dark found":
            no_dark.append(obsfile)
        elif "error" in result:
            problem_obs.append(obsfile)
            exception_log.append(result)

    # Log problems
    if not os.path.exists(orbit_subfolder_path): # Just in case this folder is full of erroring files
        os.mkdir(orbit_subfolder_path)

    finish_time_str = datetime.datetime.now().strftime('%Y-%m-%d')
    logfn = f"process_report_{startorb}-{startorb+100}_{finish_time_str}{report_append_text}.txt"
    logpath = orbit_subfolder_path + logfn

    # Calculate total problems 
    total_probs = len(problem_obs) + len(not_in_dark_key) + len(no_dark)

    with open(logpath, "w") as f:
        f.write(f"SUCCESSFUL FILES: {len(successful_obs)} / {len(this_md)}\n")
        f.write("==========================================================\n")
        for o in successful_obs:
            f.write(f"OK: {o['name']}\n")

        f.write(f"\nPROBLEM FILES: {total_probs} / {len(this_md)}\n")
        f.write("==========================================================\n")
        for o in not_in_dark_key:
            f.write(f"{o['name']}: Not in the light/dark key\n")
        for (o,e) in zip(problem_obs, exception_log):
            f.write(f"{o['name']}: {e}\n")
        for o in no_dark:
            f.write(f"{o['name']}: No valid dark found\n")
        f.write("\n\n")


    print(f"Finished {startorb}--{startorb+100}\n\n\n\n\n")

    # the idl error log is in the parent l1c folder, copy it into the proper subfolder.
    # then blank it out
    os.system(f"cp '{L1C_DIR + eln}' '{orbit_subfolder_path + eln}'" )
    open(L1C_DIR+eln, 'w').close()


if DO_WRITEOUT:
    idlproc.terminate()

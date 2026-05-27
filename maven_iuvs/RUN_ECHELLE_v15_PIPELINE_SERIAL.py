import sys
import os
import datetime
import queue
import threading
import subprocess
import pandas as pd
from astropy.io import fits
import argparse
from pathlib import Path

# statistics.py is duplicated in maven_iuvs, but it's also a base package.
# this causes problems when this script lives where maven_iuvs lives.
# Deal with it by shuffling the path info
stdlib_path = next(p for p in sys.path if 'site-packages' not in p and 'dist-packages' not in p)
if stdlib_path not in sys.path:
    sys.path.insert(0, stdlib_path)

project_dir = os.path.abspath(os.path.dirname(__file__))
if project_dir in sys.path:
    sys.path.remove(project_dir)
    sys.path.append(project_dir)

from maven_iuvs.download import get_default_data_directory
from maven_iuvs.echelle import get_dir_metadata, find_files_with_geometry, \
     downselect_data, convert_l1a_to_l1c, get_dark_from_keyfile, \
     pipe_processer, command_IDL_and_verify_done, \
     open_idl_and_compile_writel1c_script
from maven_iuvs.miscellaneous import orbit_folder, iuvs_orbno_from_fname

# ARG PARSE 
# =============================================================================
# Orbit arguments are optional IFF you have specified specific files to fit.
parser = argparse.ArgumentParser(description='Orbits to process to v15')
parser.add_argument('start_orb', type=int, nargs='?',
                    help='Start orbit (multiple of 100)')
parser.add_argument('end_orb', type=int, nargs ='?',
                    help='End orbit (multiple of 100) -- will not be included')

args = parser.parse_args()

# WORKER FUNCTIONS FOR PARALLELIZATION ========================================

def process_observation(obs_md, orbfold, ldkey, process_timestamp, clean_data_kwargs=None,
                        idl_process_kwargs=None, make_plots=False, overwrite=False,
                        save_arrays=False, fitter="dynesty", writeout=True):
    """
    Specific process for setting up and performing fits to frames within an 
    observation defined my obs_md. Keeps things tidy by returning at various 
    points, which is better than more awkward control flow if this was part of 
    obs_worker(). See that function for the arguments. 
    """
    ti = datetime.datetime.now()

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
            PLACE_FOR_FIGS = orbfold + f"{l1c_fn}" + "/plot_fits/"
            if not os.path.exists(PLACE_FOR_FIGS):
                os.makedirs(PLACE_FOR_FIGS)
            plot_kwargs = {"fig_savepath": PLACE_FOR_FIGS}
        else:
            plot_kwargs = {}

        if save_arrays:
            PLACE_FOR_ARRAYS = orbfold + f"{l1c_fn}" + "/plotted_arrays/"
            if not os.path.exists(PLACE_FOR_ARRAYS):
                os.makedirs(PLACE_FOR_ARRAYS)
        else:
            PLACE_FOR_ARRAYS = ""

        # Call the conversion 
        status = convert_l1a_to_l1c(lf, df, lfold+ln, dfold+dn,
                                    orbfold, 
                                    fitter=fitter,
                                    livepts=50,
                                    bound="multi",
                                    overwrite=overwrite,
                                    process_timestamp=process_timestamp,
                                    save_arrays=save_arrays,
                                    place_for_arrays=PLACE_FOR_ARRAYS,
                                    calibration="new", 
                                    make_plots=make_plots,
                                    clean_data_kwargs=clean_data_kwargs,                       
                                    plot_kwargs=plot_kwargs,
                                    run_writeout=writeout,
                                    idl_process_kwargs=idl_process_kwargs,
                                    )

        tf = datetime.datetime.now()

        print(f"Finished with file {obs_md['name']} in {tf-ti} seconds. ")
        print("continuing...")
        print()
        return status 

    except Exception as excep:
        return f"Caught an error: {excep}"


def _record_result(obs_md, result, shared_results):
    """
    Add the metadata dictionary for a particular observation to the shared 
    manager dictionary, so results can be written out later

    Parameters
    ----------
    obs_md : dictionary
             Dictionary containing observation metadata for a particular observation
    result : string
             Short description of the result of processing the file
    shared_results : dictionary
                     manager Dictionary that stores sorted metadata dicts
    lock : lock object
           prevents race conditions on the dictionaries
    
    Results
    ----------
    n/a
    """
    if "OK" in result:
        shared_results['OK'].append(obs_md)
    elif result=="Light not in the key":
        shared_results['no_light'].append(obs_md)
    elif result=="No valid dark found":
        shared_results['no_dark'].append(obs_md)
    elif "error" in result:
        shared_results['other'].append(obs_md)
        shared_results['other_log'].append(f"{result}")
    return shared_results


# SET UP LOOP 
# =============================================================================
def main():

    # SET UP ======================================================================
    DO_WRITEOUT = True # If True, IDL will run.
    make_plots = False
    save_arrays = False
    fitter = "dynesty" # "dynesty"
    binning = None  # "nonlinear" #  can specify nonlienar to redo those files. 
                    # Had to do this at one point due to an IDL problem.
    segment = None #"outlimb"
    if binning=="nonlinear":
        print("WARNING! Only running nonlinear files! Is that what you wanted?")
    reportext = ""  # Extra text to append to procesisng report filename
    overwrite = True
    idl_process_kwargs = {}
    # You can fill in this list if you just have a few specific files to run
    specific_files = []


    # STARTING VERSION AND FOLDER DECLARATION
    # =========================================================================
    v = "v14"
    which_l1a = {"v13": "l1a", "v14": "l1a_full_mission_reprocess"}

    IUVS_FOLD = "/home/emc/Insync/OneDrive-CU/Research/IUVS/"

    # L1c base 
    IUVS_DATA_DIR = "/media/emc/ExtremePro/IUVS/IUVS_Data/"
    L1C_DIR = IUVS_DATA_DIR + "l1c_ech_data/FMR_v15/TestingSerial/"
                # Replacement_Files # Dir for redoing files that produced 
                #                    false positives
                # DynestyAWS_2 # most recent dir.
                # Dynesty # original results, full of writeout errors.
    
    # LOAD LIGHT/DARK PAIR CSV
    # =========================================================================
    keyname = f"MASTER_LIGHT_DARK_KEY_{v}.csv"
            #input("Please type name of light/dark key to use with .csv: ")
    PF = "/home/emc/GITREPOS/maven_iuvs/maven_iuvs/ancillary/" + keyname

    print("Loading light/dark pair CSV")
    ld_pairs = pd.read_csv(PF, delimiter=",", header=0)
    if ld_pairs.empty:
        raise KeyError("dataframe is empty for some reason")

    # LOAD INDICES
    # =========================================================================
    ech_l1a_idx = get_dir_metadata(get_default_data_directory(which_l1a[v]),
                                geospatial=True)

    # Find geometry files
    lights_with_geom = find_files_with_geometry(ech_l1a_idx)

    # SELECT DATA
    # =========================================================================
    metadata_lists = []
    if not specific_files:
        if args.start_orb is None or args.end_orb is None:
            raise ValueError("Error: Please specify start and end orbits or " \
                             "specific files to run")
        orbit_folders_to_run =  list(range(args.start_orb, args.end_orb, 100))
    else:
        orbit_folders_to_run = list(set([int(iuvs_orbno_from_fname(f) - (iuvs_orbno_from_fname(f) % 100) )
                                for f in specific_files]))
    print(f"Running orbit folders {orbit_folders_to_run}")

    for so in orbit_folders_to_run:
        # Select the files for each orbit folder
        if not specific_files:
            metadata_lists.append(downselect_data(lights_with_geom,
                                                  light_dark="light",
                                                  binning=binning,
                                                  segment=segment,
                                                  orbit=[so, so+99]
                                                 )
                                )
        else:
            files_this_orbit_block = downselect_data(lights_with_geom, 
                                                     light_dark="light",
                                                     binning=binning, 
                                                     orbit=[so, so+99])
            metadata_lists.append([f for f in files_this_orbit_block \
                                   if f['name'] in specific_files])
            

        # Create the folder so we can open IDL 
        if not os.path.isdir(L1C_DIR + f'orbit{so:05}'):
            makeme = L1C_DIR + f'orbit{so:05}/'
            os.mkdir(makeme)

    print(f"Total files to process: {sum([len(m) for m in metadata_lists])}")

    # get date time here because it's hard to do in IDL
    process_timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

        
    for i, startorb in enumerate(orbit_folders_to_run):
        print(f"Now working: {startorb}")

        # Get the orbit subfolder
        this_orbfold = f"{L1C_DIR}{orbit_folder(startorb)}/"
        if not os.path.exists(this_orbfold):
            os.mkdir(this_orbfold)

        # Get files to process for this orbit: list of dictionaries
        obs_to_process = metadata_lists[i]
            
        # IDL
        # =============================================================================
        if DO_WRITEOUT: 
            # Set up the log paths to keep track of high-level output and
            # errors from IDL. o/eln = "output/error log name"
            oln = this_orbfold + f"IDLoutput_{startorb}-{startorb+100}.txt"
            eln = this_orbfold + f"IDLerrors_{startorb}-{startorb+100}.txt" 


            proc, stderr_q, stderr_thread, stdout_q, stdout_thread = \
                open_idl_and_compile_writel1c_script(this_orbfold,
                                                     output_log=oln,
                                                     err_log=eln)

            idl_process_kwargs={"open_idl": False,
                                "proc": proc,
                                "stdout_queue": stdout_q, 
                                "stderr_queue": stderr_q,
                                "stdout_thread": stdout_thread, 
                                "stderr_thread": stderr_thread}
        else:
            print("File writeout not requested, IDL will not be opened")

        # The Manager also has to keep a dictionary of high-level results
        # for each file so that the outcome can be logged by Python
        resdict_thisorb = dict({"OK": [], "no_light": [], 
                                "no_dark": [], "other": [], 
                                "other_log": []})
    

        for obs_md in obs_to_process:
            # Do the processing
            result = process_observation(obs_md, this_orbfold, ld_pairs, 
                                         process_timestamp,
                                        make_plots=make_plots, 
                                        overwrite=overwrite,
                                        save_arrays=save_arrays, 
                                        fitter=fitter, 
                                        writeout=True,
                                        clean_data_kwargs={},
                                        idl_process_kwargs=idl_process_kwargs)

            # put result in dict
            resdict_thisorb = _record_result(obs_md, result, resdict_thisorb)
        
        # end queues
        # stderr_q.join()
        # stdout_q.join()
        
        # Calculate problems that we had
        total_probs = len(resdict_thisorb['no_light']) + \
                        len(resdict_thisorb['no_dark']) + \
                        len(resdict_thisorb['other'])
        
        # Set up the log file 
        finish_time_str = datetime.datetime.now().strftime('%Y-%m-%d')
        logfn = f"process_report_{startorb}-{startorb+100}_{finish_time_str}{reportext}.txt"
        logpath = this_orbfold + logfn

        # Log the results
        with open(logpath, "w") as f:
            f.write(f"SUCCESSFUL FILES: {len(resdict_thisorb['OK'])} / {len(obs_to_process)}\n")
            f.write("==========================================================\n")
            for o in resdict_thisorb['OK']:
                expected_path = this_orbfold + o['name'].replace('l1a', 'l1c').replace('v14', 'v15')
                print(f"Python is looking for {expected_path}")
                if Path(expected_path).is_file():
                    f.write(f"OK: {o['name']}\n")
                else:
                    resdict_thisorb['other'] = resdict_thisorb['other'] + [o]
                    resdict_thisorb['other_log'] = resdict_thisorb['other_log'] \
                        + ["Python was ok but I guess IDL writeout failed"]
    
            f.write(f"\nPROBLEM FILES: {total_probs} / {len(obs_to_process)}\n")
            f.write("==========================================================\n")
            for o in resdict_thisorb['no_light']:
                f.write(f"{o['name']}: Not in the light/dark key\n")
            for o in resdict_thisorb['no_dark']:
                f.write(f"{o['name']}: No valid dark found\n")
            for (o,e) in zip(resdict_thisorb['other'], resdict_thisorb['other_log']):
                f.write(f"{o['name']}: {e}\n")
            
            f.write("\n\n")

        print(f"Finished {startorb}--{startorb+100}\n\n\n\n\n")


if __name__ == "__main__":
    main()
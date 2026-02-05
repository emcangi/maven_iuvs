import sys
import os
import datetime
import queue
import threading
import subprocess
import multiprocessing as mp
import pandas as pd
import numpy as np
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

import maven_iuvs as iuvs # look, idk why, but it breaks if this isn't here.
from maven_iuvs.download import get_default_data_directory
from maven_iuvs.echelle import get_dir_metadata, find_files_with_geometry, \
     downselect_data, convert_l1a_to_l1c, get_dark_from_keyfile, \
     pipe_processer, command_IDL_and_verify_done
from maven_iuvs.miscellaneous import orbit_folder, iuvs_orbno_from_fname, \
    iuvs_filename_to_datetime, iuvs_segment_from_fname

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


def obs_worker(process_timestamp, obs_md, orbfold, ldkey, shared_results, lock, 
               idl_process_kwargs, clean_data_kwargs, make_plots, overwrite, 
               save_arrays, fitter, writeout):
    """
    Worker function for a particular file;

    Parameters
    ----------
    obs_md : dictionary
             Dictionary of metadata for a specific observation file.
    orbfold : string
              Path to the orbit folder that obs_md results belong in.
    ldkey : Pandas dataframe
            Light/dark pair dataframe
    shared_results : multiprocessing.Manager.dict() object
                     Used to safely cache results of the processing generated
                     by the worker pool.
    
    Returns
    ----------
    None
    """
    # Process the observation: Do the fitting, construct the IDL command,
    # put the command in the IDL queue (this happens deep within, in writeout_l1c)
    result = process_observation(obs_md, orbfold, ldkey, process_timestamp,
                                 make_plots=make_plots, overwrite=overwrite,
                                 save_arrays=save_arrays, fitter=fitter, writeout=writeout,
                                 clean_data_kwargs=clean_data_kwargs,
                                 idl_process_kwargs=idl_process_kwargs)

    _record_result(obs_md, result, shared_results, lock)


def _record_result(obs_md, result, shared_results, lock):
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
    with lock:
        if "OK" in result:
            shared_results['OK'] = shared_results['OK'] + [obs_md]
        elif result=="Light not in the key":
            shared_results['no_light'] = shared_results['no_light'] + [obs_md]
        elif result=="No valid dark found":
            shared_results['no_dark'] = shared_results['no_dark'] + [obs_md]
        elif "error" in result:
            shared_results['other'] = shared_results['other'] + [obs_md]
            shared_results['other_log'] = shared_results['other_log'] + [f"{result}"]


def idl_writer(idl_cmd_q, idl_cmd, idl_pipeline_dir, idl_outlog_path, 
               idl_errlog_path):
    """
    Main target of the Process object: writes commands to the IDL subprocess.
    
    Parameters
    ----------
    idl_cmd_q : manager.Queue() instance 
                Collects and distributes the commands to be written to the IDL
                subprocess.
    idl_pipeline_dir : string
                       Location of the IDL scripts
    idl_outlog_path,  
    idl_errlog_path : string(s)
                      Paths to logs of IDL output and errors

    Returns 
    ----------
    n/a
    """
    os.chdir(idl_pipeline_dir)
    
    # Open the IDL process
    proc = subprocess.Popen(idl_cmd,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True, bufsize=1)

    # Start a queue for stderr so we can watch for script compilation success
    # This queue can be a basic queue because it's not shared between processes
    err_queue = queue.Queue()

    # start stderr thread that keeps track of IDL output
    stderr_thread = threading.Thread(target=pipe_processer, 
                                     args=(proc.stderr, idl_errlog_path, err_queue), 
                                     daemon=True)
    stderr_thread.start()

    stdout_thread = threading.Thread(target=pipe_processer, 
                                     args=(proc.stdout, idl_outlog_path), 
                                     daemon=True)
    stdout_thread.start()


    # Compile the l1c writeout script, make sure it's ready -------------------
    # Set up a queue watcher to listen for script compilation success
    compiled_msg = "% Compiled module: WRITE_L1C_FILE_FROM_PYTHON."
    compile_cmd = ".com write_l1c_file_from_python.pro\n"
    command_IDL_and_verify_done(err_queue, proc, compile_cmd, compiled_msg)

    # -------------------------------------------------------------------------
    # Now that the script is ready, IDL will collect commands and run them           
    try:
        while True:
            # Get a command from the queued list of commands and write it to IDL
            item = idl_cmd_q.get()
            # the item None indicates we are done for now
            if item is None:
                break
            proc.stdin.write(item.rstrip("\n") + "\n")
            proc.stdin.flush()
    finally:
        proc.stdin.close()
        proc.wait()
    
        # Wait for the reader thread to drain any remaining output
        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)


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
    IDL_FOLD = IUVS_FOLD + "IDL_pipeline/"
    # L1c base 
    IUVS_DATA_DIR = "/media/emc/ExtremePro/IUVS/IUVS_Data/"
    L1C_DIR = IUVS_DATA_DIR + "l1c_ech_data/FMR_v15/Replacement_Files/"
                # Replacement_Files # Dir for redoing files that produced 
                #                    false positives
                # Dynesty_AWS2 # most recent dir.
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
    clean_kwargs = {}
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

    # Set up a multiprocessing context. Spawn is safest, each child process
    # inherits only what's necessary to run the target (idl_writer).
    ctx = mp.get_context("spawn")

    with ctx.Manager() as manager:
        # Loop over orbit folders
        
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
                print("Opening IDL and loading the MAVEN environment")
                
                # Managed queue to store 'process the file commands', which 
                # will be sent to the IDL subprocess. Managed because many
                # workers have to be able to write to the queue and we need to 
                # avoid race conditions.
                idl_cmd_q = manager.Queue()

                # Set up the log paths to keep track of high-level output and
                # errors from IDL. o/eln = "output/error log name"
                oln = this_orbfold + f"IDLoutput_{startorb}-{startorb+100}.txt"
                eln = this_orbfold + f"IDLerrors_{startorb}-{startorb+100}.txt" 

                # Start the main Process object: it calls the function idl_writer,
                # which writes commands to the IDL subprocess
                IDLwriter = ctx.Process(target=idl_writer,
                                        args=(idl_cmd_q, 
                                              # Arguments for launchign IDL, 
                                              # required to make it work in a 
                                              # subprocess.
                                              ["stdbuf", "-oL", "-eL", "idl", "-quiet"], 
                                              IDL_FOLD,
                                              oln, eln))  
                IDLwriter.start()

                idl_process_kwargs={"open_idl": False,
                                    "cmd_queue": idl_cmd_q}
            else:
                print("File writeout not requested, IDL will not be opened")

            # The Manager also has to keep a dictionary of high-level results
            # for each file so that the outcome can be logged by Python
            resdict_thisorb = manager.dict({"OK": [], "no_light": [], 
                                            "no_dark": [], "other": [], 
                                            "other_log": []})
            
            # Create a lock to use with the dictionary: Locks prevent mangling
            # because multiple processes will be using the same object. 
            # See Programming Guidelines for multiprocessing: 
            # "Do not use a proxy object from more than one thread unless you 
            # protect it with a lock."
            # https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming
            lock = manager.Lock()

            # PROCESS *ALL* THE FILES!!!!!
            with ctx.Pool(processes=os.process_cpu_count()) as pool:
                # Map iterable tasks to the workers; starmap is used because 
                # multiple arguments to obs_worker are required.
                pool.starmap(obs_worker, 
                            [(process_timestamp, obs, this_orbfold, ld_pairs,
                              resdict_thisorb, lock, idl_process_kwargs,
                              clean_kwargs, make_plots, overwrite, save_arrays,
                              fitter, DO_WRITEOUT)
                            for obs in obs_to_process]
                            )
                idl_cmd_q.put(None) # Tell the queue sentinel to quit
                pool.close()
                pool.join()
            
            # Once finished, end the queues for the logs and quit the writer
            if DO_WRITEOUT:
                IDLwriter.join()

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
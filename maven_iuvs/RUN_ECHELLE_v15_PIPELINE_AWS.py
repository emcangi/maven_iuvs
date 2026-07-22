import sys
import os

# Restrict thread numbers so that BLAS doesn't spawn dozens of threads per fitting process, which tanks AWS performance
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import re
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
import cProfile
import pstats

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
     downselect_data, convert_l1a_to_l1c, get_dark_from_keyfile
from maven_iuvs.miscellaneous import orbit_folder, revision_RE


# ARG PARSE 
# =============================================================================
parser = argparse.ArgumentParser(description='Orbits to process to v15')
parser.add_argument('start_orb', type=int, 
                    help='Start orbit (multiple of 100)')
parser.add_argument('end_orb', type=int, 
                    help='End orbit (multiple of 100) -- will not be included')

args = parser.parse_args()

# WORKER FUNCTIONS FOR PARALLELIZATION ========================================

def process_observation(obs_md, orbfold, ldkey, process_timestamp, 
                        py_process_kwargs=None, idl_process_kwargs=None, 
                        clean_data_kwargs=None, make_plots=False, overwrite=False,
                        save_arrays=False, writeout=True):
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
        lp = lfold + ln
        dp = dfold + dn 
        
        # Have to mangle the pathnames due to choices made at a lower level of the
        # pipeline
        if not os.path.isfile(lp):
            # Look for the revision number...
            rXX = re.search(revision_RE, lp).group(0)
            lp = lp.replace(rXX, 'r01')
        
        if not os.path.isfile(dp):
            rXX = re.search(revision_RE, dp).group(0)
            dp = dp.replace(rXX, 'r01')
        
        # Open the fits
        lf = fits.open(lp)
        df = fits.open(dp)
        
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
        status = convert_l1a_to_l1c(lf, df, lp, dp,
                                    orbfold, 
                                    process_timestamp=process_timestamp,
                                    run_writeout=writeout,
                                    overwrite=overwrite,
                                    save_arrays=save_arrays,
                                    make_plots=make_plots,
                                    place_for_arrays=PLACE_FOR_ARRAYS,
                                    py_process_kwargs=py_process_kwargs,
                                    clean_data_kwargs=clean_data_kwargs,                       
                                    plot_kwargs=plot_kwargs,
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
               py_process_kwargs, # new
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
    result = process_observation(obs_md, orbfold, ldkey, process_timestamp,
                                 make_plots=make_plots, overwrite=overwrite,
                                 save_arrays=save_arrays, 
                                 writeout=writeout,
                                 py_process_kwargs=py_process_kwargs, 
                                 clean_data_kwargs=clean_data_kwargs,
                                 idl_process_kwargs=idl_process_kwargs)

    _record_result(obs_md, result, shared_results, lock)


def _record_result(obs_md, result, shared_results, lock):
    """Thread/process‑safe update of the shared result dict."""
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


def tee_reader(pipe, log_path: str, out_queue: queue.Queue | None = None):
    """
    Reads line‑by‑line from ``pipe`` (stdout or stderr),
    writes each line to ``log_path`` and optionally puts the line
    into ``out_queue``.
    """
    with open(log_path, "a", encoding="utf-8") as f:
        for line in iter(pipe.readline, ""):          # blocks until a line or EOF
            f.write(line)
            f.flush()
            if out_queue is not None:
                out_queue.put(line)                   # non‑blocking (unbounded queue)
    pipe.close()                                  # signal EOF to the producer


def phrase_watcher(src_queue: queue.Queue,
                   phrase: str,
                   found_event: threading.Event,
                   stop_event: threading.Event):
    """
    Consumes ``src_queue`` until ``phrase`` is observed or ``stop_event`` is set.
    When the phrase is found, ``found_event`` is set and the function returns.
    """
    while not stop_event.is_set():
        try:
            # Use a short timeout so we can react to ``stop_event`` promptly.
            line = src_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if phrase in line:
            found_event.set()
            # We still drain the queue so the tee_reader can finish cleanly.
            # (Otherwise the queue could fill up and block the tee_reader.)
            while not src_queue.empty():
                src_queue.get_nowait()
            break


def idl_writer(idl_cmd_q, idl_cmd, idl_pipeline_dir, idl_outlog_path, 
               idl_errlog_path):
    """does the heavy lifting."""
    os.chdir(idl_pipeline_dir)
    
    # Open the IDL process
    proc = subprocess.Popen(idl_cmd,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True, bufsize=1)

    # Start a queue for stderr so we can watch for script compilation success
    err_queue = queue.Queue()

    # start stderr thread that keeps track of IDL output
    errlog_thread = threading.Thread(target=tee_reader, 
                                     args=(proc.stderr, idl_errlog_path, err_queue), 
                                     daemon=True)
    errlog_thread.start()

    outlog_thread = threading.Thread(target=tee_reader, 
                                     args=(proc.stdout, idl_outlog_path), 
                                     daemon=True)
    outlog_thread.start()


    # Watcher that listens for script compilation success --------------------------------------------
    compile_phrase = "% Compiled module: WRITE_L1C_FILE_FROM_PYTHON."
    compiled_evt = threading.Event()
    stop_watcher_evt = threading.Event()
    compile_watcher = threading.Thread(
        target=phrase_watcher,
        args=(err_queue, compile_phrase, compiled_evt, stop_watcher_evt),
        daemon=True,
    )
    compile_watcher.start()

    # compile script
    proc.stdin.write(".com write_l1c_file_from_python.pro\n")
    proc.stdin.flush()

    compile_timeout = 3
    if not compiled_evt.wait(timeout=compile_timeout):
        stop_watcher_evt.set()
        raise TimeoutError(f"Compile phrase not seen within {compile_timeout}s")

    # Token found – we can stop the watcher cleanly
    stop_watcher_evt.set()
    compile_watcher.join(timeout=1)

    # ------------------------------------------------------------------------------
                    
    try:
        while True:
            item = idl_cmd_q.get()
            if item is None:
                break
            proc.stdin.write(item.rstrip("\n") + "\n")
            proc.stdin.flush()
    finally:
        proc.stdin.close()
        proc.wait()
    
        # Wait for the reader thread to drain any remaining output
        outlog_thread.join(timeout=2)
        errlog_thread.join(timeout=2)


# SET UP LOOP 
# =============================================================================
def main():

    # SET UP ======================================================================
    DO_WRITEOUT = True # If True, IDL will run.
    make_plots = False # Whether to make plots of eveyr integration
    save_arrays = False # Whether to save fitting arrays to a file 
    overwrite = True # If True, existing files will be overwritten. If False, skipped.
    binning = None  # "nonlinear" #  can specify nonlienar to redo those files. 
                    # Had to do this at one point due to an IDL problem.
    if binning=="nonlinear":
        print("WARNING! Only running nonlinear files! Is that what you wanted?")
    
    # Set nitty-gritty stuff for Python fittin ghere:
    py_process_kwargs = {"fitter": "dynesty",  #or scipy
                    "livepts": 50, # Shoot for n*2, where n = number of degrees of freedom. 50 is conservative
                    "bound": "multi", # this is a good default
                    "calibration": "v15"}  # v15 = new LSF, v14 = old LSF
    idl_process_kwargs = {}  # Will store any arguments for controlling IDL processing

    reportext = ""  # Extra text to append to processing report filename

    # STARTING VERSION AND FOLDER DECLARATION
    # =============================================================================
    l1a_version = "v14"
    l1a_foldername = {"v13": "l1a", "v14": "l1a_full_mission_reprocess"}
    IDL_FOLD = "/home/ubuntu/fake_idl_dir/" #IUVS_FOLD + "IDL_pipeline/"
    # L1c base 
    IUVS_DATA_DIR = "/MAVEN/IUVS/full_mission_reprocess/products/level1a/"
    L1C_DIR = "/MAVEN/IUVS/full_mission_reprocess/echelle_lasp/FMR_v14_l1a_to_v15_l1c/new_LSF/"

    keyname = "MASTER_LIGHT_DARK_KEY_v14_AWS.csv"
    PF = "/home/ubuntu/GITREPOS/maven_iuvs/maven_iuvs/ancillary/" + keyname

    # LOAD LIGHT/DARK PAIR CSV
    # =============================================================================
    print("Loading light/dark pair CSV")
    ld_pairs = pd.read_csv(PF, delimiter=",", header=0)
    if ld_pairs.empty:
        raise KeyError("dataframe is empty for some reason")

    # LOAD INDICES
    # =============================================================================
    metadata_dir = "/home/ubuntu/WorkingDir/"
    ech_l1a_idx = get_dir_metadata(get_default_data_directory(l1a_foldername[l1a_version]),
                                   version=l1a_version, geospatial=True, idx_dir=metadata_dir)

    # Find geometry files
    lights_with_geom = find_files_with_geometry(ech_l1a_idx)

    # SELECT DATA
    # =============================================================================
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

    print(f"Total files to process: {sum([len(m) for m in metadata_lists])}")

    # get date time here because it's hard to do in IDL
    process_timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

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
                
                # Set up queues for IDL: queue for calling the writeout script
                idl_cmd_q = manager.Queue()

                # Set up the output log path 
                oln = this_orbfold + f"IDLoutput_{startorb}-{startorb+100}.txt"
                eln = this_orbfold + f"IDLerrors_{startorb}-{startorb+100}.txt" 

                # Start the writer process
                IDLwriter = ctx.Process(target=idl_writer,
                                        args=(idl_cmd_q, 
                                            ["stdbuf", "-oL", "-eL", "idl", "-quiet"], # IDL launch command
                                            IDL_FOLD,
                                            oln,
                                            eln))   
                IDLwriter.start()

                idl_process_kwargs={"open_idl": False,
                                    "cmd_queue": idl_cmd_q}
            else:
                print("File writeout not requested, IDL will not be opened")

            # make a shared dict for storing results
            resdict_thisorb = manager.dict({"OK": [], "no_light": [], 
                                            "no_dark": [], "other": [], 
                                            "other_log": []})
            lock = manager.Lock()
            

            # PROCESS *ALL* THE FILES!!!!!
            with ctx.Pool(processes=os.cpu_count(), maxtasksperchild=10) as pool:
                pool.starmap(obs_worker, 
                            [(process_timestamp, obs, this_orbfold, ld_pairs, 
                              resdict_thisorb, lock, py_process_kwargs, 
                              idl_process_kwargs, clean_kwargs, 
                              make_plots, overwrite, save_arrays,
                              DO_WRITEOUT)
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
                    exp_path2 = expected_path.replace('r02', 'r01').replace('r03', 'r01') # really annoys me how they handle revisions for production. really does.
                    print(f"Python checking {expected_path} exists...", end="", flush=True)
                    if Path(expected_path).is_file() or Path(exp_path2).is_file():
                        print("..found")
                        f.write(f"OK: {o['name']}\n")
                    else:
                        print("...not found")
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

#profiler.disable()
#stats = pstats.Stats(profiler)
#stats.sort_stats('cumulative')
#stats.print_stats(30)

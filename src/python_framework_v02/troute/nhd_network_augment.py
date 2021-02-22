import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from functools import partial
from itertools import chain
import os
import sys
import pathlib
import argparse
import json
from tqdm import tqdm
import time
import datetime
import statistics
import netCDF4 as nc

import fsspec
import cProfile
import pstats
import re
from pstats import SortKey
from memory_profiler import profile
import requests 
import gc
import contextlib
from pympler import tracker
import psutil
import ctypes

# set the root path for output
root = pathlib.Path("../../../").resolve()
sys.path.append(os.path.join(root, "src", "python_framework_v01"))

import nhd_network_utilities_v02 as nnu
import nhd_network
import nhd_io
import network_dl

# set up a file for memory useage logging
# objMemoryUseLog = open("memory_useage.log", "w+")


def trim_memory() -> int:
  libc = ctypes.CDLL("libc.so.6")
  return libc.malloc_trim(0)
  # end trim_memory


def should_trim_memory() -> bool:
  # check if we're close to our OOM limit
  # through psutil
  process = psutil.Process(os.getpid())
  return process.memory_info().rss > MEMORY_THRESHOLD
  # end should_trim_memory


def trim_loop() -> None:
  while True:
    time.sleep(jitter(30, 60)) # jitter between 30 and 60s
    if not should_trim_memory(): 
      continue
    ret = trim_memory()
    print("trim memory result: ", ret)
    # end trim_loop


# @profile(stream=objMemoryUseLog)    # decorator for memory profiler 
def funWriteToScreenAndTextFile(strMessage):

    # print to the screen
    print(strMessage)
    
    # add (appropiate) messages to the run details text file
    objRunDetailsTextFile.writelines('{}\n'.format(strMessage))
    
    # end funWriteToScreenAndTextFile


# @profile(stream=objMemoryUseLog)    # decorator for memory profiler 
def funHmsString(sec_elapsed):
    
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)
    
    # end funHmsString


def funDumpGarbage():
    
    # Force collection
    funWriteToScreenAndTextFile('===============================')
    funWriteToScreenAndTextFile("GARBAGE:")
    funWriteToScreenAndTextFile('\n')
    gc.collect()

    funWriteToScreenAndTextFile('===============================')
    funWriteToScreenAndTextFile("{:,} GARBAGE OBJECTS:".format(len(gc.garbage)))
    funWriteToScreenAndTextFile('\n')
    
    intCounter = 0
    for x in gc.garbage:
        intCounter += 1
        # s = str(x)
        # if len(s) > 80: s = s[:77]+'...'
        funWriteToScreenAndTextFile('{:,}) {} --> {}'.format(intCounter, type(x), x))

    # end funDumpGarbage


# @profile(stream=objMemoryUseLog)    # decorator for memory profiler 
def _handle_args():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-t",
        "--threshold_length",
        help = "threshold segment length (meters)",
        dest = "threshold",
        default = 250,
        type = int,
    )
    parser.add_argument(
        "--network",
        help="Choose from among the pre-programmed supernetworks (Pocono_TEST1, Pocono_TEST2, LowerColorado_Conchos_FULL_RES, Brazos_LowerColorado_ge5, Brazos_LowerColorado_FULL_RES, Brazos_LowerColorado_Named_Streams, CONUS_ge5, Mainstems_CONUS, CONUS_Named_Streams, CONUS_FULL_RES_v20, CapeFear_FULL_RES, PNW_HU17_FULL_RES, USSEast_HU03_FULL_RES, USWest_FULL_RES)",
        dest="supernetwork",
        choices=[
            "Pocono_TEST1",
            "Pocono_TEST2",
            "LowerColorado_Conchos_FULL_RES",
            "Brazos_LowerColorado_ge5",
            "Brazos_LowerColorado_FULL_RES",
            "Brazos_LowerColorado_Named_Streams",
            "CONUS_ge5",
            "Mainstems_CONUS",
            "CONUS_Named_Streams",
            "CONUS_FULL_RES_v20",
            "CapeFear_FULL_RES",
            "Florence_FULL_RES",
            "PNW_HU17_FULL_RES",
            "USSEast_HU03_FULL_RES",
            "USWest_FULL_RES",
        ],
        default="CapeFear_FULL_RES",
    )
    parser.add_argument(
        "-p",
        "--prune",
        help="prune short headwater reaches, 1 if yes, 0 if no",
        dest="prune",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--snap",
        help="snap junctions adjacent to short reaches, 1 if yes, 0 if no",
        dest="snap",
        action="store_true",
    )
    parser.add_argument(
        "-return_original",
        "--return_original",
        help = "return an unmodified RouteLink.nc file for the specified domain",
        dest = "return_original",
        action = "store_true",
    )

    return parser.parse_args()
    
    # end _handle_args


# # @profile(stream=objMemoryUseLog)    # decorator for memory profiler 
def get_network_data(network_name):

    # keep track of function calls
    global intCounter_fun_get_network_data
    intCounter_fun_get_network_data += 1
    
    # Create directory path variable for test/input/geo, where NHD data and masks are stored
    test_folder = os.path.join(root, r"test")
    geo_input_folder = os.path.join(test_folder, r"input", r"geo")

    # Load network meta data for the Cape Fear Basin
    supernetwork = network_name

    """
    network_data = nnu.set_supernetwork_data(
        supernetwork=supernetwork, geo_input_folder=geo_input_folder
    )
    """

    # changed to updated function name in nhd_network_utilities_v02.py
    network_data = nnu.set_supernetwork_parameters(supernetwork=supernetwork, geo_input_folder=geo_input_folder)

    # if the NHDPlus RouteLink file does not exist, download it.
    if not os.path.exists(network_data["geo_file_path"]):
        filename = os.path.basename(network_data["geo_file_path"])
        network_dl.download(network_data["geo_file_path"], network_data["data_link"])

    # read-in NHD data, retain copies for viz- and full network analysis purposes
    RouteLink = nhd_io.read(network_data["geo_file_path"])

    # select only the necessary columns of geospatial data, set the DataFrame index
    cols = [v for c, v in network_data["columns"].items()]
    
    # GET THE STRAHLER ORDER DATA TOO!
    cols.append("order")

    data = nhd_io.read(network_data["geo_file_path"])
    data = data[cols]
    data = data.set_index(network_data["columns"]["key"])
    
    # mask NHDNetwork to isolate test network - full resolution Cape Fear basin, NC
    if "mask_file_path" in network_data:
        data_mask = nhd_io.read_mask(
            network_data["mask_file_path"],
            layer_string=network_data["mask_layer_string"],
        )
        data = data.filter(data_mask.iloc[:, network_data["mask_key"]], axis=0)

    # if possible, make the NHDWaterbodyComID unique, for possible speed gain
    if data['NHDWaterbodyComID'].is_unique:  # and it should be!!
        data = data.set_index('NHDWaterbodyComID')
    
    # sort index
    data = data.sort_index()

    # replace downstreams
    data = nhd_io.replace_downstreams(data, network_data["columns"]["downstream"], 0)
    
    del cols  # scoping

    return data, RouteLink, network_data
    
    # end get_network_data


# @profile(stream=objMemoryUseLog)    # decorator for memory profiler 
def network_connections(data, network_data):

    # keep track of function calls
    global intCounter_fun_network_connections
    intCounter_fun_network_connections += 1
    
    """
    Extract upstream and downstream connections between segments in network
    Args:
        data (DataFrame): Network parameter dataset, prepared
        network_data (dict): network metadata
    Returns:
        conn (dict): downstream connections
        rconn (dict): upstream connections
    """

    # extract downstream connections
    conn = nhd_network.extract_connections(data, network_data["columns"]["downstream"])

    # extract upstream connections
    rconn = nhd_network.reverse_network(conn)

    return conn, rconn
    
    # end network_connections


# @profile(stream=objMemoryUseLog)    # decorator for memory profiler 
def build_reaches(rconn):

    # keep track of function calls
    global intCounter_fun_build_reaches
    intCounter_fun_build_reaches += 1
    
    # isolate independent subnetworks
    subnets = nhd_network.reachable_network(rconn)

    # identify the segments in each subnetwork
    subreachable = nhd_network.reachable(rconn)

    # break each subnetwork into reaches
    subreaches = {}
    for tw, net in subnets.items():
        path_func = partial(nhd_network.split_at_junction, net)
        subreaches[tw] = nhd_network.dfs_decomposition(net, path_func)

    del tw, net  # scoping
    
    return subreachable, subreaches, subnets
    
    # end build_reaches


# @profile(stream=objMemoryUseLog)    # decorator for memory profiler 
def prune_headwaters(data, threshold, network_data):

    # keep track of function calls
    global intCounter_fun_prune_headwaters
    intCounter_fun_prune_headwaters += 1
    
    # initialize list to store pruned reaches
    hw_prune_list = []

    iter_count = 1
    while 1 == 1:

        # STEP 1: Find headwater reaches:
        # --------------------------------------#

        # build connections and reverse connections
        connections, rconn = network_connections(
            data.drop(index=chain.from_iterable(hw_prune_list)), network_data
        )

        # identify headwater segments
        hws = connections.keys() - chain.from_iterable(connections.values())

        # build reaches
        subreachable, subreaches, subnets = build_reaches(rconn)

        # find headwater reaches
        hw_reaches = []
        for sublists in list(subreaches.values()):
            for rch in sublists:
                for val in rch:
                    if val in hws:
                        hw_reaches.append(rch)
                        pass

        # STEP 2: identify short headwater reaches
        # --------------------------------------#
        # find headwater reaches shorter than threshold
        short_hw_reaches = []
        for rch in hw_reaches:
            if data.loc[rch, "Length"].sum() < threshold:
                short_hw_reaches.append(rch)

        # CHECK: Are there any short headwter reaches?
        # --------------------------------------#
        if len(short_hw_reaches) == 0:
            funWriteToScreenAndTextFile("no more short headwaters to prune") 

            # if no more reaches, exit while loop
            break

        # STEP 3: trim short headwater reaches
        # --------------------------------------#
        hw_junctions = {}
        for rch in short_hw_reaches:
            hw_junctions[rch[-1]] = connections[rch[-1]]

        touched = set()
        for i, (tw, jun) in enumerate(hw_junctions.items()):
            touched.add(i)

            if list(hw_junctions.values()).count(jun) > 1:
                # two short headwaters draining to the same junction

                # record reach1 as list of segments
                reach1 = short_hw_reaches[i]

                for y, (tw_y, jun_y) in enumerate(hw_junctions.items()):
                    if jun_y == jun and tw_y != tw and y not in touched:
                        # the correlary headwater reach draining to the same junction has been found
                        reach2 = short_hw_reaches[y]

                        # trim the shorter of two reaches
                        if (
                            data.loc[reach1, "Length"].sum()
                            <= data.loc[reach2, "Length"].sum()
                        ):
                            # trim reach1
                            hw_prune_list.append(reach1)
                        else:
                            # trim reach2
                            hw_prune_list.append(reach2)

            if list(hw_junctions.values()).count(jun) == 1:
                hw_prune_list.append(short_hw_reaches[i])

        funWriteToScreenAndTextFile("completed {} iterations of headwater pruning".format(iter_count))
        iter_count += 1

    data_pruned = data.drop(index=chain.from_iterable(hw_prune_list))

    return data_pruned
    
    # end prune_headwaters


# @profile(stream=objMemoryUseLog)    # decorator for memory profiler 
def snap_junctions(data, threshold, network_data):

    # keep track of function calls
    global intCounter_fun_snap_junctions
    intCounter_fun_snap_junctions += 1
    
    """
    This function snaps junctions on opposite ends of a short reach
    by forcing the lowest order upstream tributary to drain to the reach tail.
    Short reaches are defined by a user-appointed threshold length.

    For example, consider a short reach (*) stranded between two junctions:

          \  /
           \/ | 4th
       2nd  \ |
             \| /
             *|/
              |
              |

    The algoritm would select the lowest order tributary segment to the reach head,
    which in this case is a second-order drainage, and change the downstream connection
    to the reach tail. This produces the following "snapped" network config:

          \  /
           \/ |
            \ | /
             \|/
              |
              |

    Inputs
    --------------
    - data (DataFrame): NHD RouteLink data, must contain Strahler order attribute
        - this input can be either the native NHD data OR the data with pruned headwaters

    Outputs
    --------------
    - snapped_data (DataFrame): NHD RouteLink data containing no short reaches

    """
    # create a copy of the native data for snapping
    data_snapped = data.copy()

    # snap junctions
    iter_num = 1
    while 1 == 1:

        # evaluate connections
        connections, rconn = network_connections(data_snapped, network_data)

        # build reaches
        subreachable, subreaches, subnets = build_reaches(rconn)

        # build list of headwater segments
        hws = connections.keys() - chain.from_iterable(connections.values())

        funWriteToScreenAndTextFile("-----")
        funWriteToScreenAndTextFile("Finding non-headwater reaches")
        # create a list of short reaches stranded between junctions
        short_reaches = ()
        for sublists in tqdm(list(subreaches.values())):

            for rch in sublists:
                head = rch[0]

                # if reach is not a headwater
                if rconn[head] and data.loc[rch, "Length"].sum() < threshold:
                    short_reaches += tuple(rch)

        if iter_num > 1:
            funWriteToScreenAndTextFile("After iteration {}:    {:,} short reaches remain".format(iter_num - 1, len(short_reaches)))
        else:
            funWriteToScreenAndTextFile("Prior to iteration {}: {:,} short reaches existed in the network".format(iter_num, len(short_reaches)))

        # check that short reaches exist, if none - terminate process
        if len(short_reaches) == 0:
            break

        # for each short reach, snap lower order upstream trib to downstream drainage destination
        for i, rch in enumerate(short_reaches):

            # identify reach tail (downstream-most) and head (upstream-most) segments

            if type(rch) is tuple:
                tail = rch[-1]
                head = rch[0]
            else:
                tail = rch
                head = rch

            # identify segments that drain to reach head
            us_conn = rconn[head]

            # select the upstream segment to snap
            o = data_snapped.loc[us_conn, "order"].tolist()  # Strahler order
            if min(o) != max(o):
                # select the segment with lowest Strahler order
                rch_to_move = data_snapped.loc[us_conn, "order"].idxmin()
            else:
                # if all segments are same Strahler order, select the shortest
                rch_to_move = data_snapped.loc[us_conn, "Length"].idxmin()

            # snap destination is the segment that the reach tail drains to
            snap_destination = connections[tail]

            # if reach tail doesn't have a downstrem connection (it is a network tailwater)
            # then the snap desination is the ocean
            if len(snap_destination) == 0:
                snap_destination = data_snapped.loc[tail, "to"]

            # update RouteLink data with new tributary destination info
            data_snapped.loc[rch_to_move, "to"] = snap_destination

        iter_num += 1

    return data_snapped
    
    # end snap_junctions


# @profile(stream=objMemoryUseLog)    # decorator for memory profiler 
def len_weighted_av(df, var, weight):

    # function start time
    objFunStartTime = time.time()
    
    # keep track of function calls
    global intCounter_fun_len_weighted_av
    global intTotalTime_fun_len_weighted_av
    intCounter_fun_len_weighted_av += 1
        
    """
    Calculate a weighted average
    Args:
        df (DataFrame): DataFrame containing variables to be averaged and used as weights
        var (str): name of the variable to be averaged
        weight (str): name of the variable to be used as a weight
    Returns:
        x (float32): weighted average
    """

    x = (df[weight] * df[var]).sum() / df[weight].sum()
    
    # add the time taken to the cumulative time
    intTotalTime_fun_len_weighted_av += time.time() - objFunStartTime
    
    # let the user know at increments
    if int(intCounter_fun_len_weighted_av / 1000) == intCounter_fun_len_weighted_av / 1000:
        funWriteToScreenAndTextFile('--> {}:  Function <Length Weighted Average> called:  {:,} times for a total of {:,} minutes'.format(time.asctime(time.localtime(time.time())), intCounter_fun_len_weighted_av, round(intTotalTime_fun_len_weighted_av / 60, 1)))
        
    return x
    
    # end len_weighted_av


# @profile(stream=objMemoryUseLog)    # decorator for memory profiler 
def merge_parameters(to_merge):

    # function start time
    objFunStartTime = time.time()
    
    # keep track of function calls
    global intCounter_fun_merge_parameters
    global intTotalTime_fun_merge_parameters
    intCounter_fun_merge_parameters += 1
    
    """
    length-weighted averaging of channel routing parameters across merged segments
    Args:
        to_merge (DataFrame): DataFrame containing routing parameters for segments to be merged together
    Returns:
        replace (DataFrame): weighted average
    """

    data_replace = to_merge.tail(1)
    data_replace._is_copy = None

    idx = to_merge.tail(1).index


    # using function, per original alpha solution
    # ----------
    # def len_weighted_av(df, var, weight)
    #     x = (df[weight] * df[var]).sum() / df[weight].sum()
    # ----------
    # data_replace.loc[idx, "Length"] = to_merge.Length.sum()
    # data_replace.loc[idx, "n"] = len_weighted_av(to_merge, "n", "Length")
    # data_replace.loc[idx, "nCC"] = len_weighted_av(to_merge, "nCC", "Length")
    # data_replace.loc[idx, "So"] = len_weighted_av(to_merge, "So", "Length")
    # data_replace.loc[idx, "BtmWdth"] = len_weighted_av(to_merge, "BtmWdth", "Length")
    # data_replace.loc[idx, "TopWdth"] = len_weighted_av(to_merge, "TopWdth", "Length")
    # data_replace.loc[idx, "TopWdthCC"] = len_weighted_av(to_merge, "TopWdthCC", "Length")
    # data_replace.loc[idx, "MusK"] = len_weighted_av(to_merge, "MusK", "Length")
    # data_replace.loc[idx, "MusX"] = len_weighted_av(to_merge, "MusX", "Length")
    # data_replace.loc[idx, "ChSlp"] = len_weighted_av(to_merge, "ChSlp", "Length")

    
    # just do it here
    data_replace.loc[idx, 'Length'] = to_merge.Length.sum()
    data_replace.loc[idx, 'n'] = ((to_merge['Length'] * to_merge['n']).sum()) / (to_merge['Length'].sum())  # len_weighted_av(to_merge, "n", "Length")
    data_replace.loc[idx, 'nCC'] = ((to_merge['Length'] * to_merge['nCC']).sum()) / (to_merge['Length'].sum())  # len_weighted_av(to_merge, "nCC", "Length")
    data_replace.loc[idx, 'So'] = ((to_merge['Length'] * to_merge['So']).sum()) / (to_merge['Length'].sum())  # len_weighted_av(to_merge, "So", "Length")
    data_replace.loc[idx, 'BtmWdth'] = ((to_merge['Length'] * to_merge['BtmWdth']).sum()) / (to_merge['Length'].sum())  # len_weighted_av(to_merge, "BtmWdth", "Length")
    data_replace.loc[idx, 'TopWdth'] = ((to_merge['Length'] * to_merge['TopWdth']).sum()) / (to_merge['Length'].sum())  # len_weighted_av(to_merge, "TopWdth", "Length")
    data_replace.loc[idx, 'TopWdthCC'] = ((to_merge['Length'] * to_merge['TopWdthCC']).sum()) / (to_merge['Length'].sum()) # len_weighted_av(to_merge, "TopWdthCC", "Length")
    data_replace.loc[idx, 'MusK'] = ((to_merge['Length'] * to_merge['MusK']).sum()) / (to_merge['Length'].sum())  # len_weighted_av(to_merge, "MusK", "Length")
    data_replace.loc[idx, 'MusX'] = ((to_merge['Length'] * to_merge['MusX']).sum()) / (to_merge['Length'].sum())  # len_weighted_av(to_merge, "MusX", "Length")
    data_replace.loc[idx, 'ChSlp'] = ((to_merge['Length'] * to_merge['ChSlp']).sum()) / (to_merge['Length'].sum())  # len_weighted_av(to_merge, "ChSlp", "Length")
    
    # add the time taken to the cumulative time
    intTotalTime_fun_merge_parameters += time.time() - objFunStartTime
    
    # let the user know at increments
    if int(intCounter_fun_merge_parameters / 1000) == intCounter_fun_merge_parameters / 1000:
        funWriteToScreenAndTextFile('--> {}:  Function <Merge Parameters> called:  {:,} times for a total of {:,} minutes'.format(time.asctime(time.localtime(time.time())), intCounter_fun_merge_parameters, round(intTotalTime_fun_merge_parameters / 60, 1)))

    del idx, to_merge  # scoping
    
    return data_replace
    
    # merge_parameters


# @profile(stream=objMemoryUseLog)    # decorator for memory profiler 
def correct_reach_connections(data_merged):

    # function start time
    objFunStartTime = time.time()
    
    # keep track of function calls
    global intCounter_fun_correct_reach_connections
    global intTotalTime_fun_correct_reach_connections
    intCounter_fun_correct_reach_connections += 1
    
    """
    Update downstream connections ("to") for segments in a merged reach.
    Only updates *in-reach* connections.
    Args:
        data_merged (DataFrame): Routing parameters for segments in merged reach
    Returns:
        data_merged (DataFrame): Routing parameters for segments in merged reach with updated donwstream connections
    """

    for i, idx in enumerate(data_merged.index.values[0:-1]):
        data_merged.loc[idx, "to"] = data_merged.index.values[i + 1]
        
    # add the time taken to the cumulative time
    intTotalTime_fun_correct_reach_connections += time.time() - objFunStartTime
    
    # let the user know at increments
    if int(intCounter_fun_correct_reach_connections / 1000) == intCounter_fun_correct_reach_connections / 1000:
        funWriteToScreenAndTextFile('--> {}:  Function <Correct Reach Connections> called:  {:,} times for a total of {:,} minutes'.format(time.asctime(time.localtime(time.time())), intCounter_fun_correct_reach_connections, round(intTotalTime_fun_correct_reach_connections / 60, 1)))

    return data_merged
    
    # end correct_reach_connections


# @profile(stream=objMemoryUseLog)    # decorator for memory profiler 
def upstream_merge(data_merged, chop):

    # function start time
    objFunStartTime = time.time()
    
    # keep track of function calls
    global intCounter_fun_upstream_merge
    global intTotalTime_fun_upstream_merge
    intCounter_fun_upstream_merge += 1
    
    """
    Merge a short reach tail segment with upstream neighbor
    Args:
        data_merged (DataFrame): Routing parameters for segments in merged reach
        chop (list): list of merged-out segments
    Returns:
        data_merged (DataFrame): Routing parameters for segments in merged reach with updated donwstream connections
        chop (list): updated list of merged-out segments
    """

    # grab the two segments that need to be merged - simply the last two segments of the reach
    to_merge = data_merged.tail(2)

    # calculate new parameter values
    data_replace = merge_parameters(to_merge)

    # paste new parameters in to data_merged
    data_merged.loc[to_merge.tail(1).index] = data_replace

    # remove merged segments from data_merged
    data_merged = data_merged.drop(to_merge.head(1).index)

    # update "chop" list with merged-out segment IDs
    chop.append(to_merge.head(1).index.values[0])

    # add the time taken to the cumulative time
    intTotalTime_fun_upstream_merge += time.time() - objFunStartTime
    
    # let the user know at increments
    if int(intCounter_fun_upstream_merge / 1000) == intCounter_fun_upstream_merge / 1000:
        funWriteToScreenAndTextFile('--> {}:  Function <Upstream Merge> called:  {:,} times for a total of {:,} minutes'.format(time.asctime(time.localtime(time.time())), intCounter_fun_upstream_merge, round(intTotalTime_fun_upstream_merge / 60, 1)))
    
    del to_merge, data_replace  # scoping
    
    return data_merged, chop
    
    # end upstream_merge


# @profile(stream=objMemoryUseLog)    # decorator for memory profiler 
def downstream_merge(data_merged, chop, thresh):

    # function start time
    objFunStartTime = time.time()
    
    # keep track of function calls
    global intCounter_fun_downstream_merge
    global intTotalTime_fun_downstream_merge
    intCounter_fun_downstream_merge += 1
    
    """
    Merge short segments with their downstream neighbors
    Args:
        data_merged (DataFrame): Routing parameters for segments in merged reach
        chop (list): list of merged-out segments
        thresh (int): theshold reach length (meters)
    Returns:
        data_merged (DataFrame): Routing parameters for segments in merged reach with updated donwstream connections
        chop (list): updated list of merged-out segments
    """

    # find the upstream-most short segment and it's downstream connection
    idx_us = data_merged.loc[data_merged.Length < thresh].head(1).index.values[0]

    pos_idx_us = data_merged.index.get_loc(idx_us)
    idx_to = data_merged.iloc[pos_idx_us + 1].name

    # grab segments to be merged
    to_merge = data_merged.loc[[idx_us, idx_to]]

    # calculate new parameter values
    data_replace = merge_parameters(to_merge)

    # paste new parameters in to data_merged
    data_merged.loc[to_merge.tail(1).index] = data_replace

    # remove merged segments from data_merged
    data_merged = data_merged.drop(to_merge.head(1).index)

    # update "chop" list with merged-out segment IDs
    chop.append(to_merge.head(1).index.values[0])
    
    # add the time taken to the cumulative time
    intTotalTime_fun_downstream_merge += time.time() - objFunStartTime
    
    # let the user know at increments
    if int(intCounter_fun_downstream_merge / 1000) == intCounter_fun_downstream_merge / 1000:
        funWriteToScreenAndTextFile('--> {}:  Function <Downstream Merge> called:  {:,} times for a total of {:,} minutes'.format(time.asctime(time.localtime(time.time())), intCounter_fun_downstream_merge, round(intTotalTime_fun_downstream_merge / 60, 1)))
        
    del to_merge, idx_us, idx_to, data_replace  # scoping
    
    return data_merged, chop
    
    # end downstream_merge


# @profile(stream=objMemoryUseLog)    # decorator for memory profiler 
def merge_all(rch, data, chop):

    # function start time
    objFunStartTime = time.time()
    
    # keep track of function calls
    global intCounter_fun_merge_all
    global intTotalTime_fun_merge_all
    intCounter_fun_merge_all += 1
    
    """
    Merge all segments in a reach
    Args:
        rch (list): Segment indices in the reach to be merged
        data (DataFrame): Routing parameters for network containing the reach to be merged
        chop (list): list of merged-out segments
    Returns:
        data_merged (DataFrame): Routing parameters for segments in merged reach with updated donwstream connections
        chop (list): updated list of merged-out segments
    """

    # subset the model parameter data for this reach
    data_merged = data.loc[rch].copy()

    # grab the two segments that need to be merged - in this case, merge all segments!
    to_merge = data_merged.copy()

    # calculate new parameter values
    data_replace = merge_parameters(to_merge)

    # paste new parameters in to data_merged
    data_merged.loc[to_merge.tail(1).index] = data_replace

    # remove merged segments from data_merged - in this case, all but the last
    data_merged = data_merged.drop(data_merged.iloc[:-1, :].index)

    # update "chop" list with merged-out segment IDs
    chop.extend(list(to_merge.iloc[:-1, :].index))

    # add the time taken to the cumulative time
    intTotalTime_fun_merge_all += time.time() - objFunStartTime
    
    # let the user know at increments
    if int(intCounter_fun_merge_all / 1000) == intCounter_fun_merge_all / 1000:
        funWriteToScreenAndTextFile('--> {}:  Function <Merge All> called:  {:,} times for a total of {:,} minutes'.format(time.asctime(time.localtime(time.time())), intCounter_fun_merge_all, round(intTotalTime_fun_merge_all / 60, 1)))
        
    del to_merge, data_replace  # scoping
    
    return data_merged, chop
    
    # end merge_all


# @profile(stream=objMemoryUseLog)    # decorator for memory profiler 
def update_network_data(data, rch, data_merged, chop, rconn):

    # function start time
    objFunStartTime = time.time()
    
    # keep track of network data
    global intCounter_fun_update_network_data
    global intTotalTime_fun_update_network_data
    intCounter_fun_update_network_data += 1
    
    """
    Update the network routing parameter data with merged segment data
    Args:
        data (DataFrame): Routing parameters for network to be updated
        rch (list): Segment indices in the reach to be merged
        data_merged (DataFrame): Routing parameters for merged reach
    Returns:
        data (DataFrame): Updated network routing parameters
    """

    # drop the segments that disapeared with merger
    data = data.drop(chop)

    # adjust the segment data for those that remain
    data.loc[data_merged.index] = data_merged

    # update out of reach connections - these will change in the first segment was merged out
    upstreams = rconn[rch[0]]  # upstream connection of the OLD reach head

    if bool(upstreams):

        data.loc[upstreams, "to"] = data_merged.head(1).index.values[
            0
        ]  # index of NEW reach head
        
    # add the time taken to the cumulative time
    intTotalTime_fun_update_network_data += time.time() - objFunStartTime
    
    # let the user know at increments
    if int(intCounter_fun_update_network_data / 1000) == intCounter_fun_update_network_data / 1000:
        funWriteToScreenAndTextFile('--> {}:  Function <Update Network Data> called: {:,} times for a total of {:,} minutes'.format(time.asctime(time.localtime(time.time())), intCounter_fun_update_network_data, round(intTotalTime_fun_update_network_data / 60, 1)))

    del data_merged, upstreams  # scoping
    
    return data
    
    # end update_network_data


# @profile(stream=objMemoryUseLog)    # decorator for memory profiler 
def qlat_destination_compute(data_native, data_merged, merged_segments, pruned_segments, network_data):
    
    # keep track of function calls
    global intCounter_fun_qlat_destination_compute
    intCounter_fun_qlat_destination_compute += 1
    
    # build a list of all segments that need crosswalking
    if bool(list(pruned_segments)):
        segments = merged_segments + list(pruned_segments)

    else:
        segments = merged_segments

    # compute connections using native network data
    conn = nhd_network.extract_connections(
        data_native, network_data["columns"]["downstream"]
    )
    rconn = nhd_network.reverse_network(conn)

    # initialize a dictionary to store qlat destination nodes for pruned/merged segments
    qlat_destinations = {}

    for idx in segments:

        # find the segment to recieve qlats from the pruned or merged segment
        if conn[idx]:
            ds_idx = conn[idx]
            while bool(ds_idx[0] in data_merged.index) == False:
                ds_idx = conn[ds_idx[0]]

        elif rconn[idx]:
            ds_idx = rconn[idx]
            while bool(ds_idx[0] in data_merged.index) == False:
                us_idx = conn[ds_idx[0]]

        else:
            ds_idx = []

        # update the qlat destination dict
        qlat_destinations[str(idx)] = str(ds_idx)

    del segments, conn, rconn  # scoping
    
    return qlat_destinations
    
    # end qlat_destination_compute


# @profile(stream=objMemoryUseLog)    # decorator for memory profiler 
def segment_merge(data_native, data, network_data, thresh, pruned_segments):    # data_native is the native RouteLink file as a pandas dataframe

    # keep track of function calls
    global intCounter_fun_segment_merge
    intCounter_fun_segment_merge += 1

    
    global intCounter_fun_segment_merge_reaches_loop
    global intCounter_fun_segment_merge_reaches_loop_a
    global intCounter_fun_segment_merge_reaches_loop_b
    global intCounter_fun_segment_merge_reaches_loop_c
    global intTotalTime_segment_merge_reaches_loop
    
    
    # create a copy of the pruned network dataset, which will be updated with merged data
    data_merged = data.copy()

    # initialize list to store merged segment IDs
    merged_segments = []

    # build connections and reverse connections
    conn, rconn = network_connections(data, network_data)  # dictionary objects: conn & rconn

    # organize network into reaches
    subreachable, subreaches, subnets = build_reaches(rconn)

    # loop through each reach in the network
    for twi, (tw, rchs) in enumerate(subreaches.items(), 1):   # tw = tailwater

        for rch in rchs:
        
            # increment the loop counter
            intCounter_fun_segment_merge_reaches_loop += 1
            
            # this loop's start time
            objLoopStartTime = time.time()

            rch_len = data.loc[rch].Length.sum()

            ##################################################
            # orphaned short single segment reaches (a)
            ##################################################
            # if reach length is shorter than threshold and composed of a single segment
            if rch_len < thresh and len(data.loc[rch]) == 1:
                intCounter_fun_segment_merge_reaches_loop_a += 1
                continue  # do nothing

            ##################################################
            # multi segment reaches - combine into a single segment reach (b)
            ##################################################
            # if reach length is shorter than threshold and composed more than one segment
            if rch_len < thresh and len(data.loc[rch]) > 1:
            
                intCounter_fun_segment_merge_reaches_loop_b += 1

                # merge ALL reach segments into one
                chop = []
                reach_merged, chop = merge_all(rch, data, chop)

                # update network with merged reach data
                data_merged = update_network_data(data_merged, rch, reach_merged, chop, rconn)

                # update merged_segments list with merged-out segments
                merged_segments.extend(chop)

            ##################################################
            # multi segment reaches longer than threshold with some segments shorter than threshold (c)
            ##################################################
            # if reach length is longer than threshold and smallest segment length is less than threshold
            if rch_len > thresh and data.loc[rch].Length.min() < thresh:
            
                intCounter_fun_segment_merge_reaches_loop_c += 1

                # initialize data_merged - this DataFrame will be subsequently revised
                reach_merged = data.loc[rch]

                # initialize list of segments chopped from this reach
                chop_reach = []

                # so long as the shortest segment is shorter than the threshold...
                intSegmentMergeWhileCounter = 0
                while reach_merged.Length.min() < thresh:
                
                    # increment the While counter
                    intSegmentMergeWhileCounter += 1
                    # funWriteToScreenAndTextFile('----> segment merge While counter: {:,}'.format(intSegmentMergeWhileCounter))

                    # if shortest segment is the last segment in the reach - conduct an upstream merge.
                    if (reach_merged.Length.idxmin() == reach_merged.tail(1).index.values[0] and reach_merged.Length.min() < thresh):

                        # upstream merge
                        chop = []
                        reach_merged, chop = upstream_merge(reach_merged, chop)

                        # update chop_reach list with merged-out segments
                        chop_reach.extend(chop)

                    # if shortest segment is NOT the last segment in the reach - conduct a downstream merge
                    if (reach_merged.Length.idxmin() != reach_merged.tail(1).index.values[0] and reach_merged.Length.min() < thresh):

                        # downstream merge
                        chop = []
                        reach_merged, chop = downstream_merge(reach_merged, chop, thresh)

                        # update chop_reach list with merged-out segments
                        chop_reach.extend(chop)
                        
                    # -----------------------------------

                # correct segment connections within reach
                reach_merged = correct_reach_connections(reach_merged)

                # update the greater network data set
                data_merged = update_network_data(data_merged, rch, reach_merged, chop_reach, rconn)  # length weighted average
                
                # update merged_segments list with merged-out segments
                merged_segments.extend(chop_reach)  # cache of merging out, adding to the list 
                
                # testing shows that this data_merged pandas dataframe stays small (say, < 100 Mb, usually much less...)
                
            # add the time taken to the cumulative time
            intTotalTime_segment_merge_reaches_loop += time.time() - objLoopStartTime
            
            # let the user know at increments
            if int(intCounter_fun_segment_merge_reaches_loop / 1000) == intCounter_fun_segment_merge_reaches_loop / 1000:
                funWriteToScreenAndTextFile('----> {}:  Function <Segment Merge> - Reaches Loop called: {:,} times for a total of {:,} minutes'.format(time.asctime(time.localtime(time.time())), intCounter_fun_segment_merge_reaches_loop, round(intTotalTime_segment_merge_reaches_loop / 60, 1)))
                funWriteToScreenAndTextFile('----------------------->  Individual Reaches Loop called: {:,} | {:,} | {:,} times (a, b, c)'.format(intCounter_fun_segment_merge_reaches_loop_a, intCounter_fun_segment_merge_reaches_loop_b, intCounter_fun_segment_merge_reaches_loop_c))
                print('==========================')
                print('DATA_MERGED pandas dataframe info:')
                print('\n')
                print(data_merged.info())
                print('==========================')
                try:
                    print('\n')
                    print('==========================')
                    print('REACH_MERGED pandas dataframe info:')
                    print('\n')
                    print(reach_merged.info())
                    print('==========================')
                    print('\n')
                except NameError:
                    print('--')
                

    

    # create a qlateral destinations dictionary
    qlat_destinations = qlat_destination_compute(data_native, data_merged, merged_segments, pruned_segments, network_data) # edge situation of running hydrologic model - to allow for mass balance - dictionary of pieces pruned or merged and pushed back in 

    del merged_segments, reach_merged, chop_reach, chop, rch_len, thresh, pruned_segments, network_data, data_native  # scoping    

    return data_merged, qlat_destinations
    
    # end segment_merge


def main() -> None:

    # start the profiler, etc
    profiler = cProfile.Profile()
    profiler.enable()
    
    # garbage collection -  strat
    # gc.enable()
    # gc.set_debug(gc.DEBUG_LEAK)
    
    # grab the start time in a variable
    objTimeStart = time.time()  # record the start time, for the whole code loop
    strCurrentTime = time.asctime(time.localtime(time.time()))
    
    # unpack command line arguments
    args = _handle_args()
    supernetwork = args.supernetwork
    threshold = args.threshold
    prune = args.prune
    snap = args.snap
    return_original = args.return_original

    # create a text file to hold the oce run details
    filename_rundetails = 'RunDetails_{}_{}m_prune_snap_merge.txt'.format(supernetwork, str(threshold))
    dirname = ('RouteLink_{}_{}m_prune_snap_merge'.format(supernetwork, str(threshold)))
    dir_path = os.path.join(root, "test", "input", "geo", "Channels", dirname)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    global objRunDetailsTextFile  # so it works in various functions
    objRunDetailsTextFile = open(os.path.join(dir_path, filename_rundetails),'w')
    
    # cProfile / pstat dat file details
    filename_pstatdetails = 'PstatDetails_{}_{}m_prune_snap_merge.dat'.format(supernetwork, str(threshold))
    objPstatDetailsDatFile = os.path.join(dir_path, filename_pstatdetails)
    
    # set up some counters for number of times functions called, etc
    
    timPuningHeadwaters = None
    timSnapJunctions = None
    timSegmentsMerge = None
    
    global intCounter_fun_get_network_data
    intCounter_fun_get_network_data = 0
    
    global intCounter_fun_network_connections
    intCounter_fun_network_connections = 0
    
    global intCounter_fun_build_reaches
    intCounter_fun_build_reaches = 0
    
    global intCounter_fun_prune_headwaters
    intCounter_fun_prune_headwaters = 0
    
    global intCounter_fun_snap_junctions
    intCounter_fun_snap_junctions = 0
    
    global intCounter_fun_len_weighted_av
    intCounter_fun_len_weighted_av = 0
    global intTotalTime_fun_len_weighted_av
    intTotalTime_fun_len_weighted_av = 0  # seconds
    
    global intCounter_fun_merge_parameters
    intCounter_fun_merge_parameters = 0
    global intTotalTime_fun_merge_parameters
    intTotalTime_fun_merge_parameters = 0  # seconds
    
    global intCounter_fun_correct_reach_connections
    intCounter_fun_correct_reach_connections = 0
    global intTotalTime_fun_correct_reach_connections
    intTotalTime_fun_correct_reach_connections = 0
    
    global intCounter_fun_upstream_merge
    intCounter_fun_upstream_merge = 0
    global intTotalTime_fun_upstream_merge
    intTotalTime_fun_upstream_merge = 0  # seconds
    
    global intCounter_fun_downstream_merge
    intCounter_fun_downstream_merge = 0
    global intTotalTime_fun_downstream_merge
    intTotalTime_fun_downstream_merge = 0  # seconds
    
    global intCounter_fun_merge_all
    intCounter_fun_merge_all = 0
    global intTotalTime_fun_merge_all
    intTotalTime_fun_merge_all = 0  # seconds
    
    global intCounter_fun_update_network_data
    intCounter_fun_update_network_data = 0
    global intTotalTime_fun_update_network_data
    intTotalTime_fun_update_network_data = 0
    
    global intCounter_fun_qlat_destination_compute
    intCounter_fun_qlat_destination_compute = 0
    
    global intCounter_fun_segment_merge
    intCounter_fun_segment_merge = 0
    
    global intCounter_fun_segment_merge_reaches_loop
    global intCounter_fun_segment_merge_reaches_loop_a
    global intCounter_fun_segment_merge_reaches_loop_b
    global intCounter_fun_segment_merge_reaches_loop_c
    intCounter_fun_segment_merge_reaches_loop = 0
    intCounter_fun_segment_merge_reaches_loop_a = 0
    intCounter_fun_segment_merge_reaches_loop_b = 0
    intCounter_fun_segment_merge_reaches_loop_c = 0
    global intTotalTime_segment_merge_reaches_loop
    intTotalTime_segment_merge_reaches_loop = 0
    
    
    # show some initial details
    funWriteToScreenAndTextFile('================================================================')
    funWriteToScreenAndTextFile('CODE STARTING')
    funWriteToScreenAndTextFile('================================================================')
    funWriteToScreenAndTextFile('\n')
    funWriteToScreenAndTextFile('Details Of Script(s) Run:')
    funWriteToScreenAndTextFile('\n')
    funWriteToScreenAndTextFile('Script:                 nhd_network_augment.py (alpha solution with tweaks)')
    funWriteToScreenAndTextFile('Last Edit Date:         02/18/21 - GJC_Development Branch')
    funWriteToScreenAndTextFile('---------')
    funWriteToScreenAndTextFile('Time ALL Code Started:  {}'.format(strCurrentTime))
    funWriteToScreenAndTextFile('Arguments:              {}'.format(args))
    funWriteToScreenAndTextFile('Supernetwork:           {}'.format(supernetwork))
    funWriteToScreenAndTextFile('Threshold (m):          {}'.format(threshold))
    funWriteToScreenAndTextFile('Prune:                  {}'.format(prune))
    funWriteToScreenAndTextFile('Snap:                   {}'.format(snap))
    funWriteToScreenAndTextFile('Return Original:        {}'.format(return_original))
    funWriteToScreenAndTextFile('\n')


    # get network data
    funWriteToScreenAndTextFile('==========')
    funWriteToScreenAndTextFile("Extracting and Organizing Supernetwork Data...")
    data, RouteLink, network_data = get_network_data(supernetwork)
    RouteLink = RouteLink.set_index(network_data["columns"]["key"])
    funWriteToScreenAndTextFile('\n')
    
    pruned_segs = []
    
    # prune headwaters
    if prune and snap:
    
        funWriteToScreenAndTextFile('==========')
        funWriteToScreenAndTextFile("Prune, Snap, Then Merge:")
        funWriteToScreenAndTextFile('\n')
        
        funWriteToScreenAndTextFile('---')
        funWriteToScreenAndTextFile("Pruning Headwaters...")
        objTimeStartSection = time.time()  # record the start time, for this part of the code
        data_pruned = prune_headwaters(data, threshold, network_data)
        objTimeEndSection = time.time()  # record the end time, for this part of the code
        timPuningHeadwaters = funHmsString(objTimeEndSection - objTimeStartSection)
        funWriteToScreenAndTextFile('**** Pruning the headwaters took {} [HH:MM:SS.SS] ****'.format(timPuningHeadwaters))
        funWriteToScreenAndTextFile('\n')

        # identify pruned segments
        pruned_segs = list(np.setdiff1d(data.index, data_pruned.index))

        funWriteToScreenAndTextFile('---')
        funWriteToScreenAndTextFile("Snapping Junctions...")
        objTimeStartSection = time.time()  # record the start time, for this part of the code
        data_snapped = snap_junctions(data_pruned, threshold, network_data)
        objTimeEndSection = time.time()  # record the end time, for this part of the code
        timSnapJunctions = funHmsString(objTimeEndSection - objTimeStartSection)
        funWriteToScreenAndTextFile('**** Snapping Junctions took {} [HH:MM:SS.SS] ****'.format(timSnapJunctions))
        funWriteToScreenAndTextFile('\n')

        funWriteToScreenAndTextFile('---')
        funWriteToScreenAndTextFile("Merging Segments...")
        objTimeStartSection = time.time()  # record the start time, for this part of the code
        data_merged, qlat_destinations = segment_merge(data, data_snapped, network_data, threshold, pruned_segs)
        objTimeEndSection = time.time()  # record the end time, for this part of the code
        timSegmentsMerge = funHmsString(objTimeEndSection - objTimeStartSection)
        funWriteToScreenAndTextFile('**** Segment merging took {} [HH:MM:SS.SS] ****'.format(timSegmentsMerge))
        funWriteToScreenAndTextFile('\n')

    
    if snap and not prune:
        
        funWriteToScreenAndTextFile('==========')
        funWriteToScreenAndTextFile("Snap And Merge:")
        funWriteToScreenAndTextFile('\n')

        funWriteToScreenAndTextFile('---')
        funWriteToScreenAndTextFile("snapping junctions...")
        objTimeStartSection = time.time()  # record the start time, for this part of the code
        data_snapped = snap_junctions(data, threshold, network_data)
        objTimeEndSection = time.time()  # record the end time, for this part of the code
        timSnapJunctions = funHmsString(objTimeEndSection - objTimeStartSection)
        funWriteToScreenAndTextFile('**** Snapping Junctions took {} [HH:MM:SS.SS] ****'.format(timSnapJunctions))
        funWriteToScreenAndTextFile('\n')

        funWriteToScreenAndTextFile('---')
        funWriteToScreenAndTextFile("merging segments...")
        objTimeStartSection = time.time()  # record the start time, for this part of the code
        data_merged, qlat_destinations = segment_merge(data, data_snapped, network_data, threshold, pruned_segs)
        objTimeEndSection = time.time()  # record the end time, for this part of the code
        timSegmentsMerge = funHmsString(objTimeEndSection - objTimeStartSection)
        funWriteToScreenAndTextFile('**** Segment merging took {} [HH:MM:SS.SS] ****'.format(timSegmentsMerge))
        funWriteToScreenAndTextFile('\n')


    if not snap and prune:
        
        funWriteToScreenAndTextFile('==========')
        funWriteToScreenAndTextFile("Prune And Merge:")
        funWriteToScreenAndTextFile('\n')

        funWriteToScreenAndTextFile('---')
        funWriteToScreenAndTextFile("snapping junctions...")
        objTimeStartSection = time.time()  # record the start time, for this part of the code
        data_snapped = snap_junctions(data, threshold, network_data)
        objTimeEndSection = time.time()  # record the end time, for this part of the code
        timSnapJunctions = funHmsString(objTimeEndSection - objTimeStartSection)
        funWriteToScreenAndTextFile('**** Snapping Junctions took {} [HH:MM:SS.SS] ****'.format(timSnapJunctions))
        funWriteToScreenAndTextFile('\n')

        funWriteToScreenAndTextFile('---')
        funWriteToScreenAndTextFile("merging segments...")
        objTimeStartSection = time.time()  # record the start time, for this part of the code
        data_merged, qlat_destinations = segment_merge(data, data_snapped, network_data, threshold, pruned_segs)
        objTimeEndSection = time.time()  # record the end time, for this part of the code
        timSegmentsMerge = funHmsString(objTimeEndSection - objTimeStartSection)
        funWriteToScreenAndTextFile('**** Segment merging took {} [HH:MM:SS.SS] ****'.format(timSegmentsMerge))
        funWriteToScreenAndTextFile('\n')


    if not snap and not prune:
        
        funWriteToScreenAndTextFile('==========')
        funWriteToScreenAndTextFile("Just Merge:")
        funWriteToScreenAndTextFile('\n')

        funWriteToScreenAndTextFile('---')
        objTimeStartSection = time.time()  # record the start time, for this part of the code
        funWriteToScreenAndTextFile("merging segments...")
        data_merged, qlat_destinations = segment_merge(data, data, network_data, threshold, pruned_segs)
        objTimeEndSection = time.time()  # record the end time, for this part of the code
        timSegmentsMerge = funHmsString(objTimeEndSection - objTimeStartSection)
        funWriteToScreenAndTextFile('**** Segment merging took {} [HH:MM:SS.SS]'.format(timSegmentsMerge))
        funWriteToScreenAndTextFile('\n')


    # section - export various dataets
    objTimeStartSection = time.time()  # record the start time, for this part of the code
    
    # update RouteLink data
    RouteLink_edit = RouteLink.loc[data_merged.index.values]

    for (columnName, columnData) in data_merged.iteritems():
        RouteLink_edit.loc[:, columnName] = columnData

    for idx in RouteLink_edit.index:
        if RouteLink_edit.loc[idx, "to"] < 0:
            RouteLink_edit.loc[idx, "to"] = 0
            
    # convert RouteLink (edited) to geodataframe
    RouteLink_edit = gpd.GeoDataFrame(RouteLink_edit, geometry=gpd.points_from_xy(RouteLink_edit.lon, RouteLink_edit.lat),)

    # create RouteLink data (original)
    dirname = "RouteLink_" + supernetwork
    filename = "RouteLink_" + supernetwork + ".shp"
    dir_path = os.path.join(root, "test", "input", "geo", "Channels", dirname)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    RouteLink_domain = RouteLink.loc[data.index.values]
    RouteLink_domain = gpd.GeoDataFrame(RouteLink_domain, geometry=gpd.points_from_xy(RouteLink_domain.lon, RouteLink_domain.lat),)
    
    # create the file names, etc ready for export
    dirname = 'RouteLink_{}_{}m_prune_snap_merge'.format(supernetwork, threshold)
    filename = '{}.shp'.format(dirname)
    filename_cw = 'Crosswalk_{}.json'.format(str(dirname[10:]))
    filename_netcdf_modified = '{}_modified.nc'.format(str(dirname))
    filename_netcdf_original = '{}_original.nc'.format(str(dirname))
    dir_path = os.path.join(root, "test", "input", "geo", "Channels", dirname)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    # export merged data - shape file format
    funWriteToScreenAndTextFile('==========')
    funWriteToScreenAndTextFile('EXPORTING VARIOUS FILES TO: {}'.format(dir_path))
    funWriteToScreenAndTextFile('--')

    
    # save RouteLink data (modified)
    funWriteToScreenAndTextFile("1) Exporting edited RouteLink file as SHP 'file':      {}\{}".format(dir_path, filename))
    RouteLink_edit_dropped_cols1 = RouteLink_edit.drop(columns=["time", "gages"])
    RouteLink_edit_dropped_cols1.to_file(os.path.join(dir_path, filename))
    
    # export cross walk as json
    funWriteToScreenAndTextFile("2) Exporting CrossWalk file as JSON file:              {}\{}".format(dir_path, filename_cw))
    with open(os.path.join(dir_path, filename_cw), "w") as outfile:
        json.dump(qlat_destinations, outfile)
        
    # export original RouteLink as a NetCDF file
    funWriteToScreenAndTextFile("3) Exporting original RouteLink file as a NetCDFfile:  {}\{}".format(dir_path, filename_netcdf_original))
    xrRouteLink_domain = RouteLink_domain.drop(columns=['geometry']).to_xarray()
    objOriginalRouteLinkNetCDF = fsspec.open(os.path.join(dir_path, filename_netcdf_original), mode='wb')
    xrRouteLink_domain.to_netcdf(objOriginalRouteLinkNetCDF, engine='netcdf4')
    objOriginalRouteLinkNetCDF.close()

    # export edited RouteLink as a NetCDF file
    funWriteToScreenAndTextFile("4) Exporting edited RouteLink file as a NetCDFfile:    {}\{}".format(dir_path, filename_netcdf_modified))
    xrRouteLink_edit = RouteLink_edit.drop(columns=['geometry']).to_xarray()
    objModifiedRouteLinkNetCDF = fsspec.open(os.path.join(dir_path, filename_netcdf_modified), mode='wb')
    xrRouteLink_edit.to_netcdf(objModifiedRouteLinkNetCDF, engine='netcdf4')
    objModifiedRouteLinkNetCDF.close()

    # export original data
    if return_original:
        
        funWriteToScreenAndTextFile("5) Exporting original RouteLink file as SHP 'file':    {}\{}".format(dir_path, filename))
        RouteLink_domain = RouteLink_domain.drop(columns=["time", "gages"])
        RouteLink_domain.to_file(os.path.join(dir_path, filename))

    objTimeEndSection = time.time()  # record the end time, for this part of the code
    funWriteToScreenAndTextFile('--')
    funWriteToScreenAndTextFile('**** it took {} [HH:MM:SS.SS] to export the various datasets ****'.format(funHmsString(objTimeEndSection - objTimeStartSection)))
    funWriteToScreenAndTextFile('\n')

    # grab the (overall) end time in a variable
    objTimeEnd = time.time()  # record the end time total, overall
    strCurrentTime = time.asctime(time.localtime(time.time()))
    funWriteToScreenAndTextFile('---------')
    funWriteToScreenAndTextFile('Time ALL Code Finished: {}'.format(strCurrentTime))
    funWriteToScreenAndTextFile('\n')
    
    # record of function call counts
    funWriteToScreenAndTextFile('==========')
    funWriteToScreenAndTextFile('Count & Time Metrics  For Various Function Calls:')
    funWriteToScreenAndTextFile('\n')
    funWriteToScreenAndTextFile('Function: <prune_headwaters> [HH:MM:SS.SS]   {}'.format(timPuningHeadwaters))
    funWriteToScreenAndTextFile('Function: <snap_junctions>   [HH:MM:SS.SS]   {}'.format(timSnapJunctions))
    funWriteToScreenAndTextFile('Function: <segment_merge>    [HH:MM:SS.SS]   {}'.format(timSegmentsMerge))
    funWriteToScreenAndTextFile('--')
    funWriteToScreenAndTextFile('Function <get_network_data> Called:             {:>12,} times'.format(intCounter_fun_get_network_data))
    funWriteToScreenAndTextFile('Function <network_connections> Called:          {:>12,} times'.format(intCounter_fun_network_connections))
    funWriteToScreenAndTextFile('Function <build_reaches> Called:                {:>12,} times'.format(intCounter_fun_build_reaches))
    funWriteToScreenAndTextFile('Function <prune_headwaters> Called:             {:>12,} times'.format(intCounter_fun_prune_headwaters))
    funWriteToScreenAndTextFile('Function <snap_junctions> Called:               {:>12,} times'.format(intCounter_fun_snap_junctions))
    funWriteToScreenAndTextFile('Function <len_weighted_av> Called:              {:>12,} times @ {:>6,} minutes total'.format(intCounter_fun_len_weighted_av, round(intTotalTime_fun_len_weighted_av / 60, 2)))
    funWriteToScreenAndTextFile('Function <merge_parameters> Called:             {:>12,} times @ {:>6,} minutes total'.format(intCounter_fun_merge_parameters, round(intTotalTime_fun_merge_parameters / 60, 2)))
    funWriteToScreenAndTextFile('Function <correct_reach_connections> Called:    {:>12,} times @ {:>6,} minutes total'.format(intCounter_fun_correct_reach_connections, round(intTotalTime_fun_correct_reach_connections / 60, 2)))
    funWriteToScreenAndTextFile('Function <upstream_merge> Called:               {:>12,} times @ {:>6,} minutes total'.format(intCounter_fun_upstream_merge, round(intTotalTime_fun_upstream_merge / 60, 2)))
    funWriteToScreenAndTextFile('Function <downstream_merge> Called:             {:>12,} times @ {:>6,} minutes total'.format(intCounter_fun_downstream_merge, round(intTotalTime_fun_downstream_merge / 60, 2)))
    funWriteToScreenAndTextFile('Function <merge_all> Called:                    {:>12,} times @ {:>6,} minutes total'.format(intCounter_fun_merge_all, round(intTotalTime_fun_merge_all / 60, 2)))
    funWriteToScreenAndTextFile('Function <update_network_data> Called:          {:>12,} times @ {:>6,} minutes total'.format(intCounter_fun_update_network_data, round(intTotalTime_fun_update_network_data / 60, 2)))
    funWriteToScreenAndTextFile('Function <qlat_destination_compute> Called:     {:>12,} times'.format(intCounter_fun_qlat_destination_compute))
    funWriteToScreenAndTextFile('Function <segment_merge> Called:                {:>12,} times'.format(intCounter_fun_segment_merge))
    funWriteToScreenAndTextFile('Function <segment_merge> Reaches Loop Count:    {:>12,} times @ {:>6,} minutes total'.format(intCounter_fun_segment_merge_reaches_loop, round(intTotalTime_segment_merge_reaches_loop / 60, 2)))
    funWriteToScreenAndTextFile('Function <segment_merge> Reaches Loop Count A:  {:>12,} times <-- orphaned short single segment reaches)'.format(intCounter_fun_segment_merge_reaches_loop_a))
    funWriteToScreenAndTextFile('Function <segment_merge> Reaches Loop Count B:  {:>12,} times <-- multi segment reaches - combine into a single segment reach'.format(intCounter_fun_segment_merge_reaches_loop_b))
    funWriteToScreenAndTextFile('Function <segment_merge> Reaches Loop Count C:  {:>12,} times <-- multi segment reaches longer than threshold with some segments shorter than threshold'.format(intCounter_fun_segment_merge_reaches_loop_c))
    funWriteToScreenAndTextFile('\n')
    
    # close the profiler and add stats to the metric, etc file
    profiler.disable()
    stats_cumtime = pstats.Stats(profiler).strip_dirs().sort_stats('cumtime')
    stats_cumtime.dump_stats(objPstatDetailsDatFile)
    stats_ncalls = pstats.Stats(profiler).strip_dirs().sort_stats('ncalls')
    # stats.print_stats()
    
    # let the user know summary of what happened
    funWriteToScreenAndTextFile('==========')
    funWriteToScreenAndTextFile('Results Summary:')
    funWriteToScreenAndTextFile('\n')
    funWriteToScreenAndTextFile("Number of segments (as points) in original RouteLink: {:>10,}".format(len(RouteLink_domain)))
    funWriteToScreenAndTextFile("Number of segments (as points) in modified RouteLink: {:>10,}".format(len(RouteLink_edit)))
    if len(RouteLink_domain) == len(RouteLink_edit):
        funWriteToScreenAndTextFile('--> NOTE: RouteLink has the same number of points (i.e., original and modified)...')
    funWriteToScreenAndTextFile('\n')
    funWriteToScreenAndTextFile('OVERALL: It took {} [HH:MM:SS.SS] (total) to execute ALL of this code'.format(funHmsString(objTimeEnd - objTimeStart)))
    funWriteToScreenAndTextFile('\n')
    funWriteToScreenAndTextFile('Run Details Text File:                {}'.format(filename_rundetails))
    funWriteToScreenAndTextFile('cProfiler / Pstat Details DAT File:   {}'.format(filename_pstatdetails))
    funWriteToScreenAndTextFile('=========')
    funWriteToScreenAndTextFile('\n')
    
    # show the pstats data (screen only)
    print('================================================================')
    print('cProfile / pstats Summary [secreen only]:')
    print('\n')
    print('----------')
    print('Top 30 By Cumulative Time -->')
    print('\n')
    stats_cumtime.print_stats(30)
    print('\n')
    print('----------')
    print('Top 30 By Number Of Calls -->')
    print('\n')
    stats_ncalls.print_stats(30)
    print('\n')
    
    # garbage collection - end
    # gc.disable()
    
    # show the dirty garbage ~wink~
    # funWriteToScreenAndTextFile('================================================================')
    # funWriteToScreenAndTextFile('SHOW THE DIRTY GARBAGE:')
    # funWriteToScreenAndTextFile('\n')
    # funDumpGarbage()
    # funWriteToScreenAndTextFile('\n')
    
    # finish off the code
    funWriteToScreenAndTextFile('\n')
    funWriteToScreenAndTextFile('================================================================')
    funWriteToScreenAndTextFile('CODE FINISHED')
    funWriteToScreenAndTextFile('================================================================')
    
    # close the run details text file
    objRunDetailsTextFile.close()
    # objMemoryUseLog.close()
    
    # end main
    

if __name__ == "__main__":
    
    main()
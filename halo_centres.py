#!/bin/env python

import os.path
import h5py
import numpy as np
import unyt
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.gather_array as g


def gather_to_rank_zero(arr):
    """Gather the specified array on rank 0, preserving units"""
    units = arr.units
    arr = g.gather_array(arr.value)
    return unyt.unyt_array(arr, units=units)


class SOCatalogue:

    def __init__(self, comm, vr_basename, a_unit, registry, boxsize, max_halos,
                 centrals_only, halo_prop_list):
        """
        This reads in the VR catalogues and stores the halo properties in a
        dict of unyt_arrays, self.halo_arrays, on rank 0 of communicator comm.
        It also calculates the radii to read in around each halo.

        self.halo_arrays["read_radius"] contains the radius to read in about
        the potential minimum of each halo.

        self.halo_arrays["search_radius"] contains an initial guess for the
        radius we need to search to reach the required overdensity. This will
        be increased up to read_radius if necessary.

        Both read_radius and search_radius will be set to be at least as large
        as the largest physical_radius_mpc specified by the halo property
        calculations.
        """

        comm_rank = comm.Get_rank()

        # Get SWIFT's definition of physical and comoving Mpc units
        swift_pmpc = unyt.Unit("swift_mpc",       registry=registry)
        swift_cmpc = unyt.Unit(a_unit*swift_pmpc, registry=registry)
        swift_msun = unyt.Unit("swift_msun",      registry=registry)

        # Get expansion factor as a float
        a = a_unit.base_value

        # Find minimum physical radius to read in
        physical_radius_mpc = 0.0
        for halo_prop in halo_prop_list:
            physical_radius_mpc = max(physical_radius_mpc, halo_prop.physical_radius_mpc)
        physical_radius_mpc = unyt.unyt_quantity(physical_radius_mpc, units=swift_pmpc)

        # Here we need to read the centre of mass AND potential minimum:
        # The radius R_size about (Xc, Yc, Zc) contains all particles which
        # belong to the group. But we want to compute spherical overdensity
        # quantities about the potential minimum.
        datasets = ("Xcminpot", "Ycminpot", "Zcminpot", "Xc", "Yc", "Zc",
                    "R_size", "Structuretype", "ID", "npart", "SO_R_2500_rhocrit")

        # Check for single file VR output - will prefer filename without
        # extension if both are present
        vr_basename += ".properties"
        if comm_rank == 0:
            if os.path.exists(vr_basename):
                filenames = vr_basename
            else:
                filenames = vr_basename+".%(file_nr)d"
        else:
            filenames = None
        filenames = comm.bcast(filenames)

        # Read in positions and radius of each halo, distributed over all MPI ranks
        mf = phdf5.MultiFile(filenames, file_nr_dataset="Num_of_files")
        local_halo = mf.read(datasets)

        # Compute array index of each halo
        nr_local = local_halo["ID"].shape[0]
        offset = comm.scan(nr_local) - nr_local
        local_halo["index"] = np.arange(offset, offset+nr_local, dtype=int)

        # Combine positions into one array each
        local_halo["cofm"] = np.column_stack((local_halo["Xc"], local_halo["Yc"], local_halo["Zc"]))
        del local_halo["Xc"]
        del local_halo["Yc"]
        del local_halo["Zc"]
        local_halo["cofp"] = np.column_stack((local_halo["Xcminpot"], local_halo["Ycminpot"], local_halo["Zcminpot"]))
        del local_halo["Xcminpot"]
        del local_halo["Ycminpot"]
        del local_halo["Zcminpot"]

        # Extract unit information from the first file
        if comm_rank == 0:
            filename = filenames % {"file_nr" : 0}
            with h5py.File(filename, "r") as infile:
                units = dict(infile["UnitInfo"].attrs)
                siminfo = dict(infile["SimulationInfo"].attrs)
        else:
            units = None
            siminfo = None
        units, siminfo = comm.bcast((units, siminfo))

        # Compute conversion factors to comoving Mpc (no h)
        comoving_or_physical = int(units["Comoving_or_Physical"])
        length_unit_to_kpc = float(units["Length_unit_to_kpc"])
        h = float(siminfo["h_val"])
        if comoving_or_physical == 0:
            # File contains physical units with no h factor
            length_conversion = (1.0/a) * length_unit_to_kpc / 1000.0 # to comoving Mpc
        else:
            # File contains comoving 1/h units
            length_conversion = h * length_unit_to_kpc / 1000.0 # to comoving Mpc

        # Convert units and wrap in unyt_arrays
        for name in local_halo:
            dtype = local_halo[name].dtype
            if name in ("cofm", "cofp", "R_size", "SO_R_2500_rhocrit"):
                conv_fac = length_conversion
                units = swift_cmpc
            elif name in ("Structuretype", "ID", "index", "npart"):
                conv_fac = None
                units = unyt.dimensionless
            else:
                raise Exception("Unrecognized property name: "+name)
            if conv_fac is not None:
                local_halo[name] = unyt.unyt_array(local_halo[name]*conv_fac, units=units, dtype=dtype, registry=registry)
            else:
                local_halo[name] = unyt.unyt_array(local_halo[name], units=units, dtype=dtype, registry=registry)
        #
        # Compute initial search radius for each halo:
        #
        # Need to ensure that our radius about the potential minimum
        # includes all particles within r_size of the centre of mass.
        #
        # Find distance from centre of mass to centre of potential,
        # taking the periodic box into account
        dist = np.abs(local_halo["cofp"] - local_halo["cofm"])
        for dim in range(3):
            need_wrap = dist[:,dim] > 0.5*boxsize
            dist[need_wrap, dim] = boxsize - dist[need_wrap, dim]
        dist = np.sqrt(np.sum(dist**2, axis=1))

        # Store the initial search radius
        local_halo["search_radius"] = (local_halo["R_size"]*1.01 + dist)

        # Compute radius to read in about each halo:
        # this is the maximum radius we'll search to reach the required overdensity
        local_halo["read_radius"] = local_halo["search_radius"].copy()
        min_radius = 5.0*swift_cmpc
        ind = local_halo["read_radius"] < min_radius
        local_halo["read_radius"][ind] = min_radius

        # Ensure that both the initial search radius and the radius to read in
        # are >= the minimum physical radius required by property calculations
        ind = local_halo["read_radius"] < physical_radius_mpc
        local_halo["read_radius"][ind] = physical_radius_mpc
        ind = local_halo["search_radius"] < physical_radius_mpc
        local_halo["search_radius"][ind] = physical_radius_mpc

        # Discard satellites, if necessary
        if centrals_only:
            keep = local_halo["Structuretype"] == 10
            for name in local_halo:
                local_halo[name] = local_halo[name][keep,...]
        
        # Gather subhalo arrays on rank zero.
        halo = {}
        for name in local_halo:
            halo[name] = gather_to_rank_zero(local_halo[name])
        del local_halo

        # For testing: limit number of halos
        if comm_rank == 0 and max_halos > 0:
            nr_halo = 0
            for name in halo:
                nr_halo = len(halo[name])
                break
            beg = nr_halo//2
            end = beg+max_halos
            print(f"Processing halos {beg}->{end}")
            for name in halo:
                halo[name] = halo[name][beg:end,...]

        """
        if comm_rank == 0:
            idx = [11230,17508,18160,19257,19638,19652,21828,22261,22455,24917,25011,26506,26550,27021,27558,27939,28195,28858,29248,29274,29446,30010,30500,32507,32524,33184,33335,33899,34129,34196,34384,34475,34638,34733,34924,35027,35134,35656,35805,35979,36403,36501,36559,36655,36797,36807,37175,37416,37995,38319,38354,38391,38494,39054,39274,39348,39424,39531,39631,39658,39825,40020,40156,40497,40600,40663,40809,41295,41613,41763,41808,41825,42148,42213,42343,42345,42866,42963,43122,43618,43660,43753,43763,43881,43889,43998,44258,44558,44559,44752,44800,45064,45133,45217,45403,45407,45536,45569,45633,45652,45694,45866,45916,46027,46071,46116,46148,46158,46269,46300,46326,46338,46361,46380,46419,46500,46656,46705,46850,46851,46946,46971,47100,47199,47214,47615,47714,47754,47794,47833,47836,47866,47892,47918,47964,48081,48196,48299,48402,48424,48523,48539,48679,48769,48800,48981,49009,49257,49262,49441,49500,49695,49709,49822,49958,49964,50015,50079,50137,50191,50307,50361,50547,50566,50674,50728,50751,50775,50789,50829,51173,51211,51238,51342,51373,51711,51745,51836,51860,51991,52085,52109,52129,52213,52218,52268,52340,52353,52387,52419,52446,52644,52912,52947,53025,53289,53416,53535,53903,53928,54010,54085,54365,54460,54834,54849,54886,55008,55133,55496,55595,55631,55641,55680,55751,55875,55896,55979,56121,56265,56329,56402,56464,56490,56497,56505,56530,56563,56565,56627,56767,57204,57376,57390,57419,57679,57706,57767,57929,58203,58223,58342,58496,58624,58671,58774,58901,58958,58995,59090,59209,59254,59276,59307,59378,59403,59423,59453,59700,59838,59985,60024,60043,60090,60133,60168,60292,60355,60357,60455,60525,60759,60785,60789,61110,61223,61260,61541,61542,61568,61680,61712,61720,61875,61964,62051,62059,62220,62263,62353,62462,62477,62504,62535,62536,62599,62602,62612,62664,62749,62768,62798,62836,62939,62953,62957,63186,63201,63279,63358,63407,63499,63518,63593,63610,63642,63939,63993,64020,64025,64058,64115,64120,64126,64138,64145,64157,64232,64259,64336,64436,64635,64774,64814,64828,64872,64911,64928,65105,65120,65130,65165,65175,65197,65232,65234,65341,65362,65488,65695,65697,65753,65755,65864,65937,65946,66067,66082,66140,66239,66284,66424,66509,66571,66594,66600,66654,66657,66749,66757,66841,66876,67032,67161,67204,67242,67314,67406,67693,67727,67894,67947,68018,68053,68149,68281,68356,68385,68491,68539,68578,68637,68694,68717,68827,68893,68906,68968,69068,69075,69082,69107,69157,69273,69298,69315,69451,69501,69623,69653,69736,69771,69788,69904,69947,70015,70022,70082,70166,70178,70286,70301,70383,70424,70443,70543,70603,70623,70626,70780,70788,70868,70921,71064,71081,71216,71221,71375,71379,71506,71581,71583,71653,71656,71675,71810,71894,71915,71983,71985,72031,72071,72084,72225,72269,72287,72341,72363,72390,72403,72407,72432,72434,72440,72488,72497,72517,72529,72531,72605,72627,72704,72747,72760,72888,72907,72943,72965,72983,72986,72990,73049,73118,73120,73158,73182,73215,73372,73430,73435,73496,73599,73671,73699,73782,73924,74065,74067,74168,74186,74400,74457,74478,74491,74502,74512,74671,74691,74746,74758,74759,74804,74821,74880,74895,74926,74960,75022,75034,75060,75062,75141,75173,75200,75233,75257,75262,75371,75392,75418,75519,75551,75694,75699,75746,75752,75832,75860,75864,75880,76001,76098,76100,76117,76143,76179,76204,76235,76267,76291,76298,76319,76333,76358,76367,76429,76433,76475,76555,76556,76562,76588,76650,76651,76678,76694,76895,76947,77079,77089,77091,77095,77142,77196,77202,77214,77239,77244,77272,77294,77299,77305,77308,77373,77514,77593,77603,77706,77744,77837,77976,77987,78000,78010,78020,78022,78086,78101,78151,78154,78176,78219,78276,78494,78548,78561,78681,78729,78871,78881,78913,78934,78983,78985,78998,79011,79068,79113,79157,79229,79269,79305,79308,79310,79330,79471,79476,79492,79515,79547,79632,79652,79663,79678,79712,79723,79834,79905,79935,79974,79990,80058,80063,80137,80250,80304,80461,80483,80487,80541,80566,80571,80637,80643,80714,80736,80746,80748,80776,80777,80781,80783,80858,80929,80949,80960,81059,81073,81185,81254,81256,81353,81355,81356,81400,81554,81619,81663,81676,81691,81759,81793,81855,81940,81943,81963,82000,82006,82114,82123,82136,82170,82189,82281,82290,82308,82336,82346,82562,82583,82592,82598,82605,82635,82683,82691,82745,82749,82751,82752,82759,82778,82779,82819,82861,82894,82907,82908,83014,83077,83089,83091,83172,83252,83283,83287,83358,83388,83406,83411,83438,83556,83640,83656,83699,83708,83816,83901,83946,83957,83982,84029,84243,84335,84346,84352,84380,84381,84438,84500,84659,84680,84730,84754,84770,84789,84858,84881,84918,84932,84956,84958,84980,84992,85006,85007,85049,85058,85064,85066,85129,85136,85148,85169,85259,85272,85339,85342,85386,85470,85486,85583,85590,85609,85674,85824,85873,85975,86033,86105,86190,86194,86226,86250,86273,86319,86403,86413,86447,86449,86455,86536,86641,86667,86679,86695,86751,86756,86774,86844,86901,86940,86958,86974,86987,87029,87115,87152,87181,87217,87250,87274,87290,87300,87333,87350,87358,87359,87404,87421,87434,87442,87450,87489,87493,87541,87561,87633,87636,87641,87646,87657,87790,87806,87840,87882,87890,87933,87969,88037,88080,88098,88116,88134,88199,88221,88268,88282,88297,88342,88407,88430,88467,88493,88496,88558,88561,88645,88758,88808,88907,88925,88957,88962,89043,89049,89056,89074,89098,89134,89136,89224,89242,89249,89271,89325,89372,89425,89433,89464,89492,89505,89529,89537,89547,89619,89664,89673,89681,89683,89694,89777,89784,89786,89810,89831,89834,89888,89922,89951,90084,90087,90098,90234,90235,90506,90562,90566,90620,90711,90730,90731,90744,90769,90804,90861,90875,90908,90929,90953,91035,91051,91067,91074,91079,91095,91164,91167,91182,91257,91277,91338,91360,91379,91450,91489,91529,91588,91618,91626,91666,91704,91721,91735,91763,91799,91858,91876,91925,91957,91960,92005,92012,92055,92091,92100,92106,92178,92245,92285,92331,92434,92456,92458,92528,92553,92558,92580,92581,92615,92648,92651,92663,92671,92730,92757,92863,92865,93056,93085,93091,93112,93147,93149,93210,93320,93349,93359,93371,93412,93441,93450,93458,93506,93526,93539,93562,93610,93622,93654,93662,93690,93693,93695,93733,93750,93764,93767,93807,93835,93854,93856,94025,94052,94071,94088,94114,94126,94157,94176,94242,94245,94262,94270,94281,94290,94307,94318,94326,94376,94385,94395,94404,94433,94442,94504,94535,94539,94631,94663,94780,94803,94810,94835,94839,94861,94862,94930,94937,95020,95067,95081,95105,95159,95270,95331,95410,95438,95452,95486,95491,95500,95540,95546,95570,95586,95638,95698,95705,95731,95760,95842,95904,95924,95963,95964,95969,96028,96050,96108,96121,96156,96192,96324,96356,96373,96386,96430,96463,96475,96565,96583,96597,96605,96619,96687,96692,96755,96769,96775,96793,96814,96830,96877,96932,97022,97042,97044,97067,97068,97077,97094,97159,97197,97227,97279,97325,97343,97388,97418,97433,97435,97458,97529,97662,97664,97733,97734,97739,97754,97774,97800,97816,97824,97842,97857,97865,97895,97930,98061,98131,98135,98201,98202,98256,98270,98324,98332,98351,98419,98442,98560,98596,98625,98633,98637,98651,98680,98720,98725,98745,98769,98782,98783,98843,98850,98944,98986,98998,99026,99098,99146,99172,99205,99211,99245,99257,99293,99323,99424,99425,99452,99466,99478,99489,99541,99581,99588,99602,99667,99673,99733,99825,99833,99867,99928,100015,100018,100061,100294,100342,100361,100407,100485,100487,100517,100521,100537,100563,100607,100616,100644,100668,100715,100730,100744,100782,100836,100847,100866,100931,100984,101041,101080,101090,101098,101099,101144,101289,101422,101440,101444,101458,101461,101542,101556,101609,101616,101625,101669,101676,101707,101755,101855,101858,101865,101893,101975,102023,102049,102128,102163,102227,102243,102248,102253,102263,102336,102421,102458,102489,102490,102591,102643,102652,102678,102787,102809,102843,102872,102874,102913,102919,102936,102944,102951,102952,102955,102986,102993,103090,103126,103187,103276,103301,103385,103398,103441,103516,103517,103520,103541,103598,103648,103662,103729,103808,103875,103989,104002,104044,104053,104057,104097,104161,104216,104234,104239,104262,104364,104367,104397,104458,104463,104497,104504,104524,104556,104592,104608,104626,104659,104681,104684,104768,104810,104821,104828,104860,104886,104891,104895,105174,105202,105207,105264,105290,105306,105346,105374,105396,105459,105466,105504,105537,105542,105551,105556,105585,105621,105688,105730,105765,105814,105869,105905,105942,105957,105958,105992,105995,105996,106056,106135,106184,106245,106275,106276,106330,106339,106351,106353,106388,106431,106437,106464,106472,106555,106595,106596,106710,106726,106814,106839,106960,106979,107035,107100,107122,107169,107258,107286,107304,107337,107382,107471,107495,107496,107498,107522,107541,107545,107558,107559,107589,107590,107611,107628,107643,107653,107671,107677,107680,107682,107683,107691,107760,107883,107896,107902,107912,107946,107951,107969,107979,107983,108039,108080,108101,108150,108186,108212,108267,108312,108313,108322,108332,108394,108398,108436,108522,108550,108556,108574,108589,108593,108599,108612,108708,108761,108850,108857,108884,108905,108943,108951,109014,109016,109138,109149,109167,109191,109200,109216]
            for name in halo:
                halo[name] = halo[name][idx,...]
        """

        # Rank 0 stores the subhalo catalogue
        if comm_rank == 0:
            self.nr_halos = len(halo["search_radius"])
            self.halo_arrays = halo



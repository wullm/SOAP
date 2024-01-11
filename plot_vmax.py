import h5py
import matplotlib.pyplot as plt
import numpy as np

# runs = ['HYDRO_FIDUCIAL', 'DMO_FIDUCIAL']
runs = ["DMO_FIDUCIAL"]
# runs = ['HYDRO_FIDUCIAL']
# n_parts = [900, 1800]
n_parts = [900]
# n_parts = [1800]
box_size = 1000
snapshot = 9
# snapshot = 77


conc_types = ["Basic", "Soft"]
dmo_conc_types = ["DMO_Basic", "DMO_Soft"]
vmax_types = ["Basic", "Skip2", "Skip3", "Skip5", "Skip10", "Skip15", "Soft"]
for t in conc_types:
    vmax_types.append("Conc_" + t)
print("Vmax types", vmax_types)

so_data = {}
bound_data = {}
for run in runs:
    for n_part in n_parts:
        soap_filename = "/snap8/scratch/dp004/dc-mcgi1/vmax_test/"
        soap_filename += f"L{box_size:04d}N{n_part:04d}/{run}/SOAP_uncompressed/"
        soap_filename += f"halo_properties_{snapshot:04d}.hdf5"

        with h5py.File(soap_filename, "r") as file:
            SO = file["SO"]["200_crit"]
            Npart = SO["NumberOfDarkMatterParticles"][()]
            Npart += SO["NumberOfGasParticles"][()]
            Npart += SO["NumberOfStarParticles"][()]
            so_data[(run, n_part)] = {
                "mass": SO["TotalMass"][()],
                "conc": SO["Concentration"][()],
                "dmo_conc": SO["DarkMatterConcentration"][()],
                # 'vmax': SO['MaximumCircularVelocity'][()],
                # 'vmax_soft': SO['MaximumCircularVelocitySoft'][()],
                "Ndm": SO["NumberOfDarkMatterParticles"][()],
                "Npart": Npart,
                "r200": SO["SORadius"][()],
                "stellar_mass": SO["StellarMass"][()],
                "gas_mass": SO["GasMass"][()],
                "bh_mass": SO["BlackHolesDynamicalMass"][()],
                "central": SO["TotalMass"][()] != 0,
            }

            bound = file["BoundSubhaloProperties"]
            Npart = bound["NumberOfDarkMatterParticles"][()]
            Npart += bound["NumberOfGasParticles"][()]
            Npart += bound["NumberOfStarParticles"][()]
            Npart += bound["NumberOfBlackHoleParticles"][()]
            bound_data[(run, n_part)] = {
                "mass": bound["TotalMass"][()],
                "vmax": bound["MaximumCircularVelocity"][()],
                "vmax_soft": bound["MaximumCircularVelocitySoft"][()],
                "r_vmax": bound["MaximumCircularVelocityRadius"][()],
                "Ndm": bound["NumberOfDarkMatterParticles"][()],
                "Npart": Npart,
                "stellar_mass": bound["StellarMass"][()],
                "gas_mass": bound["GasMass"][()],
                "bh_mass": bound["BlackHolesDynamicalMass"][()],
                "central": SO["TotalMass"][()] != 0,
            }

for (run, n_part), data in bound_data.items():
    print(run, n_part)
    not_close = np.logical_not(np.isclose(data['vmax'], data['vmax_soft']))
    mask = (data["vmax"] < data["vmax_soft"]) & not_close
    print("vmax < vmax_soft:", np.sum(mask) / mask.shape[0])
    mask = (data["vmax"] > data["vmax_soft"]) & not_close
    print("vmax > vmax_soft:", np.sum(mask) / mask.shape[0])
    print('Max r_vmax when vmax > vmax_soft:', np.max(data['r_vmax'][mask]))

exit()

#### These plots won't work since previously I was saving multiple definitions of
#### concentration/vmax in the same variable

###### Plotting high concentration objects
for npart_lim in []:
    # for npart_lim in [100]:
    all_data = so_data
    for (run, n_part), data in all_data.items():

        # Define masks
        mask = data["Npart"] > npart_lim

        # Load unmasked data
        baryon_frac = data["gas_mass"] + data["stellar_mass"] + data["bh_mass"]
        baryon_frac /= data["mass"]
        gas_frac = data["gas_mass"] / data["mass"]
        stellar_frac = data["stellar_mass"] / data["mass"]
        bh_frac = data["bh_mass"] / data["mass"]

        # Loop through and plot different properties
        for frac, xlabel, figname, lim in [
            (baryon_frac, "Baryon mass fraction", "baryon", 0.3),
            (stellar_frac, "Stellar mass fraction", "stellar", 0.3),
            (gas_frac, "Gas mass fraction", "gas", 0.3),
            (bh_frac, "Black hole mass fraction", "bh", 0.03),
        ]:

            fig, ax = plt.subplots(1)
            bins = np.linspace(0, lim, 20)
            mids = (bins[:-1] + bins[1:]) / 2

            h = np.histogram(frac[mask], bins=bins)[0].astype("float64")
            ax.plot(mids, h, label="All")

            for concentration_limit in [100, 1000]:
                conc_mask = mask & (data["conc"][:, 0] > concentration_limit)
                if np.sum(conc_mask) == 0:
                    continue
                h = np.histogram(frac[conc_mask], bins=bins)[0].astype("float64")
                ax.plot(mids, h, label=f"c>{concentration_limit}")

            ax.legend()
            # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("N")
            sim_name = f"L{box_size}N{n_part}_{run}"
            ax.set_title(f"{sim_name} {npart_lim} particle limit")

            plt.tight_layout()

            plt.savefig(f"{sim_name}_N{npart_lim}_{figname}_frac_dist.png", dpi=200)
            plt.close()


###### Plotting concentration histograms
for npart_lim in []:
    # for npart_lim in [20, 100, 300, 1000]:
    # for npart_lim in [100]:
    all_data = so_data
    for (run, n_part), data in all_data.items():
        mask = data["Npart"] > npart_lim
        # TODO: How many objects have this problem?
        mask &= np.all(data["conc"] != 0, axis=1)
        mask &= np.all(data["dmo_conc"] != 0, axis=1)
        conc = np.log10(data["conc"][mask])
        dmo_conc = np.log10(data["dmo_conc"][mask])

        fig, ax = plt.subplots(1)
        # low = min(np.min(conc), np.min(dmo_conc))
        # upp = max(np.max(conc), np.max(dmo_conc))
        # bins = np.linspace(low*0.99, upp*1.01, 21)
        bins = np.linspace(-1, 3, 41)
        mids = (bins[:-1] + bins[1:]) / 2

        for i in range(conc.shape[1]):
            arr = conc[:, i]
            h = np.histogram(arr, bins=bins)[0].astype("float64")
            std = np.std(arr)
            ls = "--" if "Soft" in conc_types[i] else "-"
            label = f"{conc_types[i]}\n$\mu=${np.median(arr):.2g}\n$\sigma=${std:.2g}\n"
            ax.plot(mids, h, label=label, ls=ls)

        if "HYDRO" in run:
            for i in range(dmo_conc.shape[1]):
                arr = dmo_conc[:, i]
                std = np.std(arr)
                h = np.histogram(arr, bins=bins)[0].astype("float64")
                ls = "--" if "Soft" in dmo_conc_types[i] else "-"
                label = f"{dmo_conc_types[i]}\n$\mu=${np.median(arr):.2g}\n$\sigma=${std:.2g}\n"
                ax.plot(mids, h, label=label, ls=ls)

        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.set_yscale("log")
        ax.set_xlabel("log(Concentration)")
        ax.set_ylabel("N")
        sim_name = f"L{box_size}N{n_part}_{run}"
        ax.set_title(f"{sim_name} {npart_lim} particle limit")

        plt.tight_layout()

        plt.savefig(f"{sim_name}_N{npart_lim}_conc_hist.png", dpi=200)
        plt.close()


##### Plot Vmax functions

# SO or bound
# all_data = so_data
# parts_used = 'SO'
all_data = bound_data
parts_used = "Bound"

bin_width = 0.05
bins = np.arange(1, 5, bin_width)
mids = (bins[:-1] + bins[1:]) / 2
volume = box_size ** 3
assert len(runs) == 1  # Only plot DMO or HYDRO

# Define which definitions of vmax to plot
# types_to_plot = np.arange(9)
# types_to_plot = [0, 1, 3, 6]
types_to_plot = [1, 3, 6]

# Calculate mass limits using lowest resolution simulation
if runs[0] == "DMO_FIDUCIAL":
    print("Determine mass bins")
    data = all_data[(runs[0], np.min(n_parts))]
    for npart_lim in [20, 30, 100, 300, 1000]:
        mask = npart_lim == data["Npart"]
        low = np.min(data["mass"][mask])
        print(f"{npart_lim} {np.log10(low):.4g}")

# Define mass limits
mass_limits = [
    (12.0, 20),
    (12.2, 30),
    (12.7, 100),
    (13.2, 300),
    (13.7, 1000),
    # (14.2, 3000),
]

for low_lim, approx_part in mass_limits:
    fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    # Use the softened vmax from the highest resolution run as the base
    base_data = all_data[(runs[0], np.max(n_parts))]
    base_x = base_data["vmax"][:, 6]
    base_mask = (10 ** low_lim < base_data["mass"]) & (
        base_data["mass"] < 10 ** (low_lim + 0.2)
    )
    # base_mask &= base_data['central']
    base_mask &= np.logical_not(base_data["central"])
    base_x = np.log10(base_x[(base_x != 0) & base_mask])
    base_n = np.histogram(base_x, bins=bins)[0].astype("float64") / (volume * bin_width)

    print(f"Mass lim: {low_lim}, (~{approx_part} particles)")
    for (run, n_part), data in all_data.items():
        for i in types_to_plot:
            calc_type = vmax_types[i]
            x = data["vmax"][:, i]
            mask = x != 0
            mask &= (10 ** low_lim < data["mass"]) & (
                data["mass"] < 10 ** (low_lim + 0.2)
            )

            # TODO: Also change base_mask above
            # mask &= data['central']
            # halo_type = 'Central'

            mask &= np.logical_not(data["central"])
            halo_type = "Satellite"

            n_halo = np.sum(mask)
            x = np.log10(x[mask])
            n = np.histogram(x, bins=bins)[0].astype("float64") / (volume * bin_width)
            p = axs[0].plot(
                mids[n != 0], n[n != 0], "-", label=f"N{n_part} {calc_type}"
            )
            color = p[0].get_color()

            mask = (n != 0) & (base_n != 0)
            axs[1].plot(mids[mask], n[mask] / base_n[mask], "-", color=color)
        print(f"{run} N{n_part}: {n_halo} halos")
    axs[0].legend(loc="lower left", ncol=2)
    axs[0].set_yscale("log")
    axs[0].set_title(f"{run} $10^{{{low_lim}}}$ mass limit, {parts_used} {halo_type}")
    axs[1].set_xlabel("log($V_{max}$)")
    axs[1].set_ylim(0.5, 1.5)
    plt.subplots_adjust(hspace=0)
    plt.savefig(f"{run}_{parts_used}_{halo_type}_M{low_lim}_vmax_function.png", dpi=200)
    plt.close()


exit()
##### Old, when I had multiple ouputs for vmax

all_data = {}
for run in runs:
    for n_part in n_parts:
        soap_filename = "/snap8/scratch/dp004/dc-mcgi1/vmax_test/"
        soap_filename += f"L{box_size:04d}N{n_part:04d}/{run}/SOAP_uncompressed/"
        soap_filename += f"halo_properties_{snapshot:04d}.hdf5"

        r_vmax = {}
        vmax = {}
        with h5py.File(soap_filename, "r") as file:
            SO = file["SO"]["200_crit"]
            for n_skip in n_skips:
                r_vmax[f"nskip{n_skip}"] = SO[f"R_vmax_nskip{n_skip}"][()]
                vmax[f"nskip{n_skip}"] = SO[f"Vmax_nskip{n_skip}"][()]
            r_vmax["assumenfw"] = SO[f"R_vmax_assumenfw"][()]
            vmax["assumenfw"] = SO[f"Vmax_assumenfw"][()]

            Npart = SO["NumberOfDarkMatterParticles"][()]
            Npart += SO["NumberOfGasParticles"][()]
            Npart += SO["NumberOfStarParticles"][()]
            Npart += SO["NumberOfBlackHoleParticles"][()]

            all_data[(run, n_part)] = {
                "mass": SO["TotalMass"][()],
                "conc": SO["Concentration"][()],
                "dmo_conc": SO["DarkMatterConcentration"][()],
                "r_vmax": r_vmax,
                "vmax": vmax,
                "Ndm": SO["NumberOfDarkMatterParticles"][()],
                "Npart": Npart,
                "r200": SO["SORadius"][()],
            }

##### Plot concentration distribution
# low = float('inf')
# upp = float(0)
# for (run, n_part), data in all_data.items():
#     mask = data['Npart'] > 20
#     conc = np.log10(data['conc'][mask])
#     low = min(low, np.min(conc))
#     upp = max(upp, np.max(conc))
#     print(np.min(conc))
#     print(np.max(conc))
# fig, ax = plt.subplots(1)
# bins = np.linspace(low*0.99, upp*1.01, 21)
# mids = (bins[:-1] + bins[1:]) / 2
# for (run, n_part), data in all_data.items():
#     mask = data['Npart'] > 20
#     conc = np.log10(data['conc'][mask])
#     ax.hist(conc,
#             bins=bins,
#             label=f'N{n_part} {run}',
#             histtype='step', linewidth=2, density=True)
# ax.legend()
# ax.set_yscale('log')
# ax.set_xlabel('log(Concentration)')
# ax.set_ylabel('pdf')
# plt.savefig('conc_hist.png', dpi=200)
# plt.close()


##### Print info about halos
# for (run, n_part), data in all_data.items():
#     conc = data['conc']
#     mass = data['mass']
#     mask = (conc > 1000) & (data['Npart'] > 20)
#     print(run, n_part, f'{np.sum(mask)/mask.shape[0]:.3g}')
#     print(np.arange(mask.shape[0])[mask])
#     print('conc', data['conc'][mask])
#     print('Ndm', data['Ndm'][mask])
#     print('Npart', data['Npart'][mask])
#     print('r200', data['r200'][mask])
#     print('r_vmaxassumenfw', data['r_vmax']['assumenfw'][mask])
#     print('vmaxassumenfw', data['vmax']['assumenfw'][mask])
#     if np.sum(mask):
#         print(np.max(data['vmax']['assumenfw'][mask]))
#     print(np.sort(mass[mask])[-5:])
#     print()


##### Plot Vmax functions
# TODO Set bins
# bin_width = 0.05
# # bins = np.arange(1.5, 3.5, bin_width)
# bins = np.arange(1, 5, bin_width)
# mids = (bins[:-1] + bins[1:]) / 2
# volume = box_size**3
# assert len(runs) == 1
# fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
# ax = axs[0]
# base_x = all_data[(runs[0], np.max(n_parts))]['vmax']['nskip10']
# base_x = np.log10(base_x[base_x != 0])
# base_n = np.histogram(base_x, bins=bins)[0].astype('float64')
# base_n /= (volume * bin_width)
# for (run, n_part), data in all_data.items():
#     for calc_type, x in data['vmax'].items():
#         mask = (x != 0)
#         # TODO Remove
#         mask &= (data['Npart'] > 20)
#         # if 'assume' in calc_type:
#             # continue
#         x = np.log10(x[mask])
#         n = np.histogram(x, bins=bins)[0].astype('float64')
#         n /= (volume * bin_width)
#         p = ax.plot(mids[n!= 0], n[n!=0], '-', label=f'N{n_part} {calc_type}')
#         color = p[0].get_color()

#         mask = (n != 0) & (base_n != 0)
#         axs[1].plot(mids[mask], n[mask]/base_n[mask], '-', color=color)
# ax.legend(loc='lower left', ncol=2)
# ax.set_yscale('log')
# ax.set_title(run)
# axs[1].set_xlabel('log($V_{max}$)')
# axs[1].set_ylim(0.8, 1.2)
# plt.subplots_adjust(hspace=0)
# plt.savefig(f'vmax_function.png', dpi=200)
# plt.close()


##### Plot Vmax distribution for different mass bins
# for (low, upp), vmax_bins in [
#         ((12, 12.1), np.linspace(2.1, 2.7, 20)),
#         ((13, 13.1), np.linspace(2.4, 2.8, 20)),
#         ((14, 14.1), np.linspace(2.75, 2.95, 20)),
#     ]:
#     fig, ax = plt.subplots(1)
#     for (run, n_part), data in all_data.items():
#         mask = (10**low < data['mass']) & (data['mass'] < 10**upp)
#         for calc_type, x in data['vmax'].items():
#             nonzero = x[mask] != 0
#             ax.hist(np.log10(x[mask][nonzero]),
#                     bins=vmax_bins,
#                     label=f'N{n_part} {calc_type} ({np.sum(nonzero)/nonzero.shape[0]:.2g})',
#                     histtype='step', density=True)
#         ax.legend()

#     ax.set_xlabel('log($V_{max}$)')
#     ax.set_title(f'{run}, $10^{{{low}}} < M_{{200}} < 10^{{{upp}}}$')
#     plt.savefig(f'{low}_vmax.png', dpi=200)
#     plt.close()


##### Plot concentration mass relation
# bins = np.arange(13, 15.51, 0.5)
# mids = (bins[:-1] + bins[1:]) / 2
# fig, ax = plt.subplots(1)
# for (run, n_part), data in all_data.items():
#     v = []
#     for i in range(mids.shape[0]):
#         mask = (10**bins[i] < data['mass']) & (data['mass'] < 10**bins[i+1])
#         v.append(np.mean(data['conc'][mask]))
#         # v.append(np.median(data['conc'][mask]))
#     ax.plot(10**mids, v, label=f'{run} N{n_part} Concentration')

#     v = []
#     for i in range(mids.shape[0]):
#         mask = (10**bins[i] < data['mass']) & (data['mass'] < 10**bins[i+1])
#         v.append(np.mean(data['dmo_conc'][mask]))
#     ax.plot(10**mids, v, label=f'{run} N{n_part} DMO_Concentration')

# ax.set_xscale('log')
# ax.set_xlabel('$M_{200c}$')
# ax.set_ylabel('$c_{Wang}$')
# ax.set_xlim(3e12, 3e15)
# ax.set_ylim(0, 10)
# ax.legend()
# plt.savefig('conc.png', dpi=200)
# plt.close()

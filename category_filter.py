#!/bin/env python

from property_table import PropertyTable


class CategoryFilter:
    """
    Filter used to determine whether properties need to be calculated for a
    certain halo or not.

    This decision is always based on the number of particles in the 6D FOF
    group, and requires the calculation of FOFSubhaloProperties for each halo.
    """

    def __init__(
        self,
        Ngeneral=100,
        Ngas=50,
        Ndm=100,
        Nstar=50,
        Nbaryon=100,
        dmo=False,
    ):
        self.Ngeneral = Ngeneral
        self.Ngas = Ngas
        self.Ndm = Ndm
        self.Nstar = Nstar
        self.Nbaryon = Nbaryon
        self.dmo = dmo

    def get_filters_direct(self, Ngas, Ndm, Nstar, Nbh):
        return {
            "basic": True,
            "general": Ngas + Ndm + Nstar + Nbh > self.Ngeneral,
            "gas": Ngas > self.Ngas,
            "dm": Ndm > self.Ndm,
            "star": Nstar > self.Nstar,
            "baryon": Ngas + Nstar > self.Nbaryon,
            "DMO": self.dmo,
        }

    def get_filters(self, halo_result):
        Ndm = halo_result[
            f"FOFSubhaloProperties/{PropertyTable.full_property_list['Ndm'][0]}"
        ][0].value
        if self.dmo:
            Ngas = 0
            Nstar = 0
            Nbh = 0
        else:
            Ngas = halo_result[
                f"FOFSubhaloProperties/{PropertyTable.full_property_list['Ngas'][0]}"
            ][0].value
            Nstar = halo_result[
                f"FOFSubhaloProperties/{PropertyTable.full_property_list['Nstar'][0]}"
            ][0].value
            Nbh = halo_result[
                f"FOFSubhaloProperties/{PropertyTable.full_property_list['Nbh'][0]}"
            ][0].value
        return self.get_filters_direct(Ngas, Ndm, Nstar, Nbh)

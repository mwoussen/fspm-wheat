# -*- coding: latin-1 -*-

import os

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from alinea.adel.adel_dynamic import AdelDyn
from alinea.adel.echap_leaf import echap_leaves

from fspmwheat import caribu_facade
from fspmwheat import cnwheat_facade
from fspmwheat import elongwheat_facade
from fspmwheat import farquharwheat_facade
from fspmwheat import growthwheat_facade
from fspmwheat import senescwheat_facade
from fspmwheat import fspmwheat_facade

from cnwheat import tools as cnwheat_tools
from cnwheat import simulation as cnwheat_simulation

"""
    test_cnwheat
    ~~~~~~~~~~~~

    Test the CN-Wheat model.

    You must first install :mod:`cnwheat` (and add it to your PYTHONPATH)
    before running this script with the command `python`.

    To get a coverage report of :mod:`cnwheat`, use the command:
    `nosetests --with-coverage --cover-package=cnwheat test_cnwheat.py`.

    CSV files must contain only ASCII characters and ',' as separator.

    :copyright: Copyright 2014-2016 INRA-ECOSYS, see AUTHORS.
    :license: TODO, see LICENSE for details.

    .. seealso:: Barillot et al. 2016.
"""

random.seed(1234)
np.random.seed(1234)

# Number of seconds in 1 hour
HOUR_TO_SECOND_CONVERSION_FACTOR = 3600

# the precision to use for quantitative comparison test
PRECISION = 4


def test_run(overwrite_desired_data=False):
    # ---------------------------------------------
    # ----- CONFIGURATION OF THE SIMULATION -------
    # ---------------------------------------------

    # -- INPUTS CONFIGURATION --

    # Path of the directory which contains the inputs of the model
    INPUTS_DIRPATH = 'inputs'

    # Name of the CSV files which describes the initial state of the system
    AXES_INITIAL_STATE_FILENAME = 'axes_initial_state.csv'
    ORGANS_INITIAL_STATE_FILENAME = 'organs_initial_state.csv'
    HIDDENZONES_INITIAL_STATE_FILENAME = 'hiddenzones_initial_state.csv'
    ELEMENTS_INITIAL_STATE_FILENAME = 'elements_initial_state.csv'
    SOILS_INITIAL_STATE_FILENAME = 'soils_initial_state.csv'
    METEO_FILENAME = 'meteo_Ljutovac2002.csv'

    # Read the inputs from CSV files and create inputs dataframes
    inputs_dataframes = {}
    for inputs_filename in (AXES_INITIAL_STATE_FILENAME,
                            ORGANS_INITIAL_STATE_FILENAME,
                            HIDDENZONES_INITIAL_STATE_FILENAME,
                            ELEMENTS_INITIAL_STATE_FILENAME,
                            SOILS_INITIAL_STATE_FILENAME):
        inputs_dataframe = pd.read_csv(os.path.join(INPUTS_DIRPATH, inputs_filename))
        inputs_dataframes[inputs_filename] = inputs_dataframe.replace({np.nan: None})

    # -- SIMULATION PARAMETERS --
    START_TIME = 0
    SIMULATION_LENGTH = 24
    PLANT_DENSITY = {1: 410}

    # define the time step in hours for each simulator
    CARIBU_TIMESTEP = 4
    SENESCWHEAT_TIMESTEP = 2
    FARQUHARWHEAT_TIMESTEP = 2
    ELONGWHEAT_TIMESTEP = 1
    GROWTHWHEAT_TIMESTEP = 1
    CNWHEAT_TIMESTEP = 1

    # Name of the CSV files which contains the meteo data
    meteo = pd.read_csv(os.path.join(INPUTS_DIRPATH, METEO_FILENAME), index_col='t')

    # -- OUTPUTS CONFIGURATION --

    # Path of the directory which contains the inputs of the model
    OUTPUTS_DIRPATH = 'outputs'

    # Name of the CSV files which will contain the outputs of the model
    DESIRED_AXES_OUTPUTS_FILENAME = 'desired_axes_outputs.csv'
    DESIRED_ORGANS_OUTPUTS_FILENAME = 'desired_organs_outputs.csv'
    DESIRED_HIDDENZONES_OUTPUTS_FILENAME = 'desired_hiddenzones_outputs.csv'
    DESIRED_ELEMENTS_OUTPUTS_FILENAME = 'desired_elements_outputs.csv'
    DESIRED_SOILS_OUTPUTS_FILENAME = 'desired_soils_outputs.csv'
    ACTUAL_AXES_OUTPUTS_FILENAME = 'actual_axes_outputs.csv'
    ACTUAL_ORGANS_OUTPUTS_FILENAME = 'actual_organs_outputs.csv'
    ACTUAL_HIDDENZONES_OUTPUTS_FILENAME = 'actual_hiddenzones_outputs.csv'
    ACTUAL_ELEMENTS_OUTPUTS_FILENAME = 'actual_elements_outputs.csv'
    ACTUAL_SOILS_OUTPUTS_FILENAME = 'actual_soils_outputs.csv'

    # create empty dataframes to shared data between the models
    shared_axes_inputs_outputs_df = pd.DataFrame()
    shared_organs_inputs_outputs_df = pd.DataFrame()
    shared_hiddenzones_inputs_outputs_df = pd.DataFrame()
    shared_elements_inputs_outputs_df = pd.DataFrame()
    shared_soils_inputs_outputs_df = pd.DataFrame()

    # define lists of dataframes to store the inputs and the outputs of the models at each step.
    axes_all_data_list = []
    organs_all_data_list = []  # organs which belong to axes: roots, phloem, grains
    hiddenzones_all_data_list = []
    elements_all_data_list = []
    soils_all_data_list = []

    all_simulation_steps = []  # to store the steps of the simulation

    # -- ADEL and MTG CONFIGURATION --

    # read adelwheat inputs at t0
    adel_wheat = AdelDyn(seed=1, scene_unit='m', leaves=echap_leaves(xy_model='Soissons_byleafclass'))
    g = adel_wheat.load(dir=INPUTS_DIRPATH)

    # ---------------------------------------------
    # ----- CONFIGURATION OF THE FACADES -------
    # ---------------------------------------------

    # -- ELONGWHEAT (created first because it is the only facade to add new metamers) --
    # Initial states
    elongwheat_hiddenzones_initial_state = inputs_dataframes[HIDDENZONES_INITIAL_STATE_FILENAME][
        elongwheat_facade.converter.HIDDENZONE_TOPOLOGY_COLUMNS + [i for i in elongwheat_facade.simulation.HIDDENZONE_INPUTS if i in
                                                                   inputs_dataframes[HIDDENZONES_INITIAL_STATE_FILENAME].columns]].copy()
    elongwheat_elements_initial_state = inputs_dataframes[ELEMENTS_INITIAL_STATE_FILENAME][
        elongwheat_facade.converter.ELEMENT_TOPOLOGY_COLUMNS + [i for i in elongwheat_facade.simulation.ELEMENT_INPUTS if i in
                                                                inputs_dataframes[ELEMENTS_INITIAL_STATE_FILENAME].columns]].copy()
    elongwheat_axes_initial_state = inputs_dataframes[AXES_INITIAL_STATE_FILENAME][
        elongwheat_facade.converter.AXIS_TOPOLOGY_COLUMNS + [i for i in elongwheat_facade.simulation.AXIS_INPUTS if i in inputs_dataframes[AXES_INITIAL_STATE_FILENAME].columns]].copy()

    phytoT = os.path.join(INPUTS_DIRPATH, 'phytoT.csv')

    # Update some parameters
    update_cnwheat_parameters = {'SL_ratio_d': 0.25}

    # Facade initialisation
    elongwheat_facade_ = elongwheat_facade.ElongWheatFacade(g,
                                                            ELONGWHEAT_TIMESTEP * HOUR_TO_SECOND_CONVERSION_FACTOR,
                                                            elongwheat_axes_initial_state,
                                                            elongwheat_hiddenzones_initial_state,
                                                            elongwheat_elements_initial_state,
                                                            shared_axes_inputs_outputs_df,
                                                            shared_hiddenzones_inputs_outputs_df,
                                                            shared_elements_inputs_outputs_df,
                                                            adel_wheat, phytoT, update_cnwheat_parameters)

    # -- CARIBU --
    caribu_facade_ = caribu_facade.CaribuFacade(g,
                                                shared_elements_inputs_outputs_df,
                                                adel_wheat)

    # -- SENESCWHEAT --
    # Initial states
    senescwheat_roots_initial_state = inputs_dataframes[ORGANS_INITIAL_STATE_FILENAME].loc[inputs_dataframes[ORGANS_INITIAL_STATE_FILENAME]['organ'] == 'roots'][
        senescwheat_facade.converter.ROOTS_TOPOLOGY_COLUMNS +
        [i for i in senescwheat_facade.converter.SENESCWHEAT_ROOTS_INPUTS if i in inputs_dataframes[ORGANS_INITIAL_STATE_FILENAME].columns]].copy()

    senescwheat_elements_initial_state = inputs_dataframes[ELEMENTS_INITIAL_STATE_FILENAME][
        senescwheat_facade.converter.ELEMENTS_TOPOLOGY_COLUMNS +
        [i for i in senescwheat_facade.converter.SENESCWHEAT_ELEMENTS_INPUTS if i in inputs_dataframes[ELEMENTS_INITIAL_STATE_FILENAME].columns]].copy()

    senescwheat_axes_initial_state = inputs_dataframes[AXES_INITIAL_STATE_FILENAME][
        senescwheat_facade.converter.AXES_TOPOLOGY_COLUMNS +
        [i for i in senescwheat_facade.converter.SENESCWHEAT_AXES_INPUTS if i in inputs_dataframes[AXES_INITIAL_STATE_FILENAME].columns]].copy()

    # Update some parameters
    update_cnwheat_parameters = {'AGE_EFFECT_SENESCENCE': 10000}

    # Facade initialisation
    senescwheat_facade_ = senescwheat_facade.SenescWheatFacade(g,
                                                               SENESCWHEAT_TIMESTEP * HOUR_TO_SECOND_CONVERSION_FACTOR,
                                                               senescwheat_roots_initial_state,
                                                               senescwheat_axes_initial_state,
                                                               senescwheat_elements_initial_state,
                                                               shared_organs_inputs_outputs_df,
                                                               shared_axes_inputs_outputs_df,
                                                               shared_elements_inputs_outputs_df, update_cnwheat_parameters)

    # -- FARQUHARWHEAT --
    # Initial states
    farquharwheat_elements_initial_state = inputs_dataframes[ELEMENTS_INITIAL_STATE_FILENAME][
        farquharwheat_facade.converter.ELEMENT_TOPOLOGY_COLUMNS +
        [i for i in farquharwheat_facade.converter.FARQUHARWHEAT_ELEMENTS_INPUTS if i in inputs_dataframes[ELEMENTS_INITIAL_STATE_FILENAME].columns]].copy()

    farquharwheat_axes_initial_state = inputs_dataframes[AXES_INITIAL_STATE_FILENAME][
        farquharwheat_facade.converter.AXIS_TOPOLOGY_COLUMNS +
        [i for i in farquharwheat_facade.converter.FARQUHARWHEAT_AXES_INPUTS if i in inputs_dataframes[AXES_INITIAL_STATE_FILENAME].columns]].copy()

    # Facade initialisation
    farquharwheat_facade_ = farquharwheat_facade.FarquharWheatFacade(g,
                                                                     farquharwheat_elements_initial_state,
                                                                     farquharwheat_axes_initial_state,
                                                                     shared_elements_inputs_outputs_df)

    # -- GROWTHWHEAT --
    # Initial states
    growthwheat_hiddenzones_initial_state = inputs_dataframes[HIDDENZONES_INITIAL_STATE_FILENAME][
        growthwheat_facade.converter.HIDDENZONE_TOPOLOGY_COLUMNS +
        [i for i in growthwheat_facade.simulation.HIDDENZONE_INPUTS if i in inputs_dataframes[HIDDENZONES_INITIAL_STATE_FILENAME].columns]].copy()

    growthwheat_elements_initial_state = inputs_dataframes[ELEMENTS_INITIAL_STATE_FILENAME][
        growthwheat_facade.converter.ELEMENT_TOPOLOGY_COLUMNS +
        [i for i in growthwheat_facade.simulation.ELEMENT_INPUTS if i in inputs_dataframes[ELEMENTS_INITIAL_STATE_FILENAME].columns]].copy()

    growthwheat_root_initial_state = inputs_dataframes[ORGANS_INITIAL_STATE_FILENAME].loc[inputs_dataframes[ORGANS_INITIAL_STATE_FILENAME]['organ'] == 'roots'][
        growthwheat_facade.converter.ROOT_TOPOLOGY_COLUMNS +
        [i for i in growthwheat_facade.simulation.ROOT_INPUTS if i in inputs_dataframes[ORGANS_INITIAL_STATE_FILENAME].columns]].copy()

    growthwheat_axes_initial_state = inputs_dataframes[AXES_INITIAL_STATE_FILENAME][
        growthwheat_facade.converter.AXIS_TOPOLOGY_COLUMNS +
        [i for i in growthwheat_facade.simulation.AXIS_INPUTS if i in inputs_dataframes[AXES_INITIAL_STATE_FILENAME].columns]].copy()

    # Update some parameters
    update_cnwheat_parameters = {'VMAX_ROOTS_GROWTH_PREFLO': 0.02885625}

    # Facade initialisation
    growthwheat_facade_ = growthwheat_facade.GrowthWheatFacade(g,
                                                               GROWTHWHEAT_TIMESTEP * HOUR_TO_SECOND_CONVERSION_FACTOR,
                                                               growthwheat_hiddenzones_initial_state,
                                                               growthwheat_elements_initial_state,
                                                               growthwheat_root_initial_state,
                                                               growthwheat_axes_initial_state,
                                                               shared_organs_inputs_outputs_df,
                                                               shared_hiddenzones_inputs_outputs_df,
                                                               shared_elements_inputs_outputs_df,
                                                               shared_axes_inputs_outputs_df, update_cnwheat_parameters)

    # -- CNWHEAT --
    # Initial states
    cnwheat_organs_initial_state = inputs_dataframes[ORGANS_INITIAL_STATE_FILENAME][
        [i for i in cnwheat_facade.cnwheat_converter.ORGANS_VARIABLES if i in inputs_dataframes[ORGANS_INITIAL_STATE_FILENAME].columns]].copy()

    cnwheat_hiddenzones_initial_state = inputs_dataframes[HIDDENZONES_INITIAL_STATE_FILENAME][
        [i for i in cnwheat_facade.cnwheat_converter.HIDDENZONE_VARIABLES if i in inputs_dataframes[HIDDENZONES_INITIAL_STATE_FILENAME].columns]].copy()

    cnwheat_elements_initial_state = inputs_dataframes[ELEMENTS_INITIAL_STATE_FILENAME][
        [i for i in cnwheat_facade.cnwheat_converter.ELEMENTS_VARIABLES if i in inputs_dataframes[ELEMENTS_INITIAL_STATE_FILENAME].columns]].copy()

    cnwheat_soils_initial_state = inputs_dataframes[SOILS_INITIAL_STATE_FILENAME][
        [i for i in cnwheat_facade.cnwheat_converter.SOILS_VARIABLES if i in inputs_dataframes[SOILS_INITIAL_STATE_FILENAME].columns]].copy()

    # Update some parameters
    update_cnwheat_parameters = {'roots': {'K_AMINO_ACIDS_EXPORT': 3E-5,
                                           'K_NITRATE_EXPORT': 1E-6}}

    # Facade initialisation
    cnwheat_facade_ = cnwheat_facade.CNWheatFacade(g,
                                                   CNWHEAT_TIMESTEP * HOUR_TO_SECOND_CONVERSION_FACTOR,
                                                   PLANT_DENSITY,
                                                   update_cnwheat_parameters,
                                                   cnwheat_organs_initial_state,
                                                   cnwheat_hiddenzones_initial_state,
                                                   cnwheat_elements_initial_state,
                                                   cnwheat_soils_initial_state,
                                                   shared_axes_inputs_outputs_df,
                                                   shared_organs_inputs_outputs_df,
                                                   shared_hiddenzones_inputs_outputs_df,
                                                   shared_elements_inputs_outputs_df,
                                                   shared_soils_inputs_outputs_df)
    # -- FSPMWHEAT --
    # Facade initialisation
    fspmwheat_facade_ = fspmwheat_facade.FSPMWheatFacade(g)

    # Update geometry
    adel_wheat.update_geometry(g)

    # ---------------------------------------------
    # -----      RUN OF THE SIMULATION      -------
    # ---------------------------------------------

    for t_caribu in range(START_TIME, SIMULATION_LENGTH, CARIBU_TIMESTEP):
        # run Caribu
        PARi = meteo.loc[t_caribu, ['PARi_MA4']].iloc[0]
        DOY = meteo.loc[t_caribu, ['DOY']].iloc[0]
        hour = meteo.loc[t_caribu, ['hour']].iloc[0]
        PARi_next_hours = meteo.loc[range(t_caribu, t_caribu + CARIBU_TIMESTEP), ['PARi']].sum().values[0]

        if (t_caribu % CARIBU_TIMESTEP == 0) and (PARi_next_hours > 0):
            run_caribu = True
        else:
            run_caribu = False

        caribu_facade_.run(run_caribu, energy=PARi, DOY=DOY, hourTU=hour, latitude=48.85, sun_sky_option='sky', heterogeneous_canopy=True, plant_density=PLANT_DENSITY[1])

        for t_senescwheat in range(t_caribu, t_caribu + CARIBU_TIMESTEP, SENESCWHEAT_TIMESTEP):
            # run SenescWheat
            senescwheat_facade_.run()

            # Run the rest of the model if the plant is alive
            for t_farquharwheat in range(t_senescwheat, t_senescwheat + SENESCWHEAT_TIMESTEP, FARQUHARWHEAT_TIMESTEP):
                # get the meteo of the current step
                Ta, ambient_CO2, RH, Ur = meteo.loc[t_farquharwheat, ['air_temperature_MA2', 'ambient_CO2_MA2', 'humidity_MA2', 'Wind_MA2']]

                # run FarquharWheat
                farquharwheat_facade_.run(Ta, ambient_CO2, RH, Ur)

                for t_elongwheat in range(t_farquharwheat, t_farquharwheat + FARQUHARWHEAT_TIMESTEP, ELONGWHEAT_TIMESTEP):
                    # run ElongWheat
                    Tair, Tsoil = meteo.loc[t_elongwheat, ['air_temperature', 'soil_temperature']]
                    elongwheat_facade_.run(Tair, Tsoil, option_static=False)

                    # Update geometry
                    adel_wheat.update_geometry(g)

                    for t_growthwheat in range(t_elongwheat, t_elongwheat + ELONGWHEAT_TIMESTEP, GROWTHWHEAT_TIMESTEP):
                        # run GrowthWheat
                        growthwheat_facade_.run()

                        for t_cnwheat in range(t_growthwheat, t_growthwheat + GROWTHWHEAT_TIMESTEP, CNWHEAT_TIMESTEP):
                            if t_cnwheat > 0:
                                # run CNWheat
                                Tair = meteo.loc[t_elongwheat, 'air_temperature']
                                Tsoil = meteo.loc[t_elongwheat, 'soil_temperature']
                                cnwheat_facade_.run(Tair, Tsoil)

                            # append outputs at current step to global lists
                            axes_outputs, elements_outputs, hiddenzones_outputs, organs_outputs, soils_outputs = fspmwheat_facade_.build_outputs_df_from_MTG()

                            all_simulation_steps.append(t_cnwheat)
                            axes_all_data_list.append(axes_outputs)
                            organs_all_data_list.append(organs_outputs)
                            hiddenzones_all_data_list.append(hiddenzones_outputs)
                            elements_all_data_list.append(elements_outputs)
                            soils_all_data_list.append(soils_outputs)

    # compare actual to desired outputs at each scale level (an exception is raised if the test failed)
    for (outputs_df_list,
         desired_outputs_filename,
         actual_outputs_filename,
         index_columns,
         state_variables_names) \
            in ((axes_all_data_list, DESIRED_AXES_OUTPUTS_FILENAME,
                 ACTUAL_AXES_OUTPUTS_FILENAME, cnwheat_simulation.Simulation.AXES_T_INDEXES, cnwheat_simulation.Simulation.AXES_STATE),
                (organs_all_data_list, DESIRED_ORGANS_OUTPUTS_FILENAME, ACTUAL_ORGANS_OUTPUTS_FILENAME,
                 cnwheat_simulation.Simulation.ORGANS_T_INDEXES, cnwheat_simulation.Simulation.ORGANS_STATE),
                (hiddenzones_all_data_list, DESIRED_HIDDENZONES_OUTPUTS_FILENAME, ACTUAL_HIDDENZONES_OUTPUTS_FILENAME,
                 cnwheat_simulation.Simulation.HIDDENZONE_T_INDEXES, cnwheat_simulation.Simulation.HIDDENZONE_STATE),
                (elements_all_data_list, DESIRED_ELEMENTS_OUTPUTS_FILENAME, ACTUAL_ELEMENTS_OUTPUTS_FILENAME,
                 cnwheat_simulation.Simulation.ELEMENTS_T_INDEXES, cnwheat_simulation.Simulation.ELEMENTS_STATE),
                (soils_all_data_list, DESIRED_SOILS_OUTPUTS_FILENAME, ACTUAL_SOILS_OUTPUTS_FILENAME,
                 cnwheat_simulation.Simulation.SOILS_T_INDEXES, cnwheat_simulation.Simulation.SOILS_STATE)):
        outputs_df = pd.concat(outputs_df_list, keys=all_simulation_steps, sort=False)
        outputs_df.reset_index(0, inplace=True)
        outputs_df.rename({'level_0': 't'}, axis=1, inplace=True)
        outputs_df = outputs_df.reindex(index_columns + outputs_df.columns.difference(index_columns).tolist(), axis=1, copy=False)
        outputs_df = outputs_df.loc[:, state_variables_names]  # compare only the values of the compartments
        cnwheat_tools.compare_actual_to_desired(OUTPUTS_DIRPATH, outputs_df, desired_outputs_filename,
                                                actual_outputs_filename, precision=PRECISION, overwrite_desired_data=overwrite_desired_data)


def compare_to_vegetative_stages(simulation_data_dir, PLANT_DENSITY):
    # -----------------------------------------------------------------------------------------
    # ----- Test outputs against reference results of vegetative_stages example ---------------
    # - Reference results are slightly diffrent from those presented in Gauthier et al. (2020)-
    # -----------------------------------------------------------------------------------------
    reference_path = r'Vegetative stages outputs'

    current_path = os.getcwd()
    simulation_data_path = os.path.abspath(os.path.join(current_path, os.pardir, 'example', simulation_data_dir))

    OUTPUTS_DIRPATH = 'outputs'
    POSTPROCESSING_DIRPATH = 'postprocessing'

    # Name of the CSV files which will contain the outputs of the model
    AXES_OUTPUTS_FILENAME = 'axes_outputs.csv'
    HIDDENZONES_OUTPUTS_FILENAME = 'hiddenzones_outputs.csv'
    # Name of the CSV files which will contain the postprocessing of the model
    AXES_POSTPROCESSING_FILENAME = 'axes_postprocessing.csv'
    ORGANS_POSTPROCESSING_FILENAME = 'organs_postprocessing.csv'
    HIDDENZONES_POSTPROCESSING_FILENAME = 'hiddenzones_postprocessing.csv'
    ELEMENTS_POSTPROCESSING_FILENAME = 'elements_postprocessing.csv'

    GRAPHS_DIRPATH = 'graphs_validation'

    with PdfPages(os.path.join(GRAPHS_DIRPATH, 'validation_graphs.pdf')) as pdf_pages:
        # -- Lmax
        # Current simulation
        simulation_res = pd.read_csv(os.path.join(simulation_data_path, OUTPUTS_DIRPATH, HIDDENZONES_OUTPUTS_FILENAME))
        simulation_res_df = simulation_res[(simulation_res['axis'] == 'MS') & (simulation_res['plant'] == 1) & ~np.isnan(simulation_res.leaf_Lmax)].copy()
        last_value_idx = simulation_res_df.groupby(['metamer'])['t'].transform(max) == simulation_res_df['t']
        simulation_res_df = simulation_res_df[last_value_idx].copy()

        # Reference simulation
        reference_res = pd.read_csv(os.path.join(reference_path, 'outputs', HIDDENZONES_OUTPUTS_FILENAME))
        reference_res_df = reference_res[(reference_res['axis'] == 'MS') & (reference_res['plant'] == 1) & ~np.isnan(reference_res.leaf_Lmax)].copy()
        last_value_idx = reference_res_df.groupby(['metamer'])['t'].transform(max) == reference_res_df['t']
        reference_res_df = reference_res_df[last_value_idx].copy()

        # Observed data
        data_obs = pd.read_csv(os.path.join(reference_path, 'inputs', 'Ljutovac2002.csv'))
        bchmk = data_obs
        bchmk = bchmk.loc[bchmk.metamer >= min(simulation_res_df.metamer)]

        # figure
        fig, (ax, ax2) = plt.subplots(2, figsize=(10, 10))

        ax.set_xlim((int(min(simulation_res_df.metamer) - 1), int(max(simulation_res_df.metamer) + 1)))
        ax.set_ylim(ymin=0, ymax=np.nanmax(list(simulation_res_df['leaf_Lmax'] * 100 * 1.05) + list(bchmk['leaf_Lmax'] * 1.05)))

        simulation_leaf_Lmax = simulation_res_df[['metamer', 'leaf_Lmax']].drop_duplicates()
        reference_leaf_Lmax = reference_res_df[['metamer', 'leaf_Lmax']].drop_duplicates()

        line1 = ax.plot(simulation_leaf_Lmax.metamer, simulation_leaf_Lmax['leaf_Lmax'] * 100, color='c', marker='o')
        line2 = ax.plot(bchmk.metamer, bchmk['leaf_Lmax'], color='orange', marker='o', linestyle='None')
        line3 = ax.plot(reference_leaf_Lmax.metamer, reference_leaf_Lmax['leaf_Lmax'] * 100, color='r', marker='o')

        ax.set_ylabel('leaf_Lmax' + ' (cm)')
        ax.set_title('leaf_Lmax')
        ax.legend((line1[0], line2[0], line3[0]), ('Current simulation', 'Ljutovac 2002', 'Reference simulation'), loc=2)

        # Second plot
        ax2.plot(simulation_leaf_Lmax['leaf_Lmax'] * 100, reference_leaf_Lmax['leaf_Lmax'] * 100, marker='o', linestyle='None')
        lims = [np.min([ax2.get_xlim(), ax2.get_ylim()]), np.max([ax2.get_xlim(), ax2.get_ylim()])]
        ax2.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

        absError = (simulation_leaf_Lmax['leaf_Lmax'].reset_index() - reference_leaf_Lmax['leaf_Lmax'].reset_index())['leaf_Lmax']
        SE = np.square(absError)  # squared errors
        MSE = np.mean(SE)  # mean squared errors
        RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
        Rsquared = 1.0 - (np.var(absError) / np.var(reference_leaf_Lmax['leaf_Lmax']))
        ax2.text(13, 45, 'RMSE = %0.2f' % RMSE)
        ax2.text(13, 42, 'R-squared = %0.2f' % Rsquared)

        ax2.set_ylabel('Reference leaf_Lmax (cm)')
        ax2.set_xlabel('Simulation leaf_Lmax (cm)')
        ax2.set_xlim(lims)
        ax2.set_ylim(lims)
        plt.savefig(os.path.join(GRAPHS_DIRPATH, 'leaf_Lmax' + '.PNG'))
        plt.show()
        pdf_pages.savefig(fig)
        plt.close()

        # -- Phyllochron
        # Current simulation
        simulation_df_SAM = pd.read_csv(os.path.join(simulation_data_path, OUTPUTS_DIRPATH, AXES_OUTPUTS_FILENAME))
        simulation_df_SAM = simulation_df_SAM[simulation_df_SAM['axis'] == 'MS']
        simulation_df_hz = pd.read_csv(os.path.join(simulation_data_path, POSTPROCESSING_DIRPATH, HIDDENZONES_POSTPROCESSING_FILENAME))
        simulation_grouped_df = simulation_df_hz[simulation_df_hz['axis'] == 'MS'].groupby(['plant', 'metamer'])[['t', 'leaf_is_emerged']]
        simulation_leaf_emergence = {}

        for group_name, data in simulation_grouped_df:
            plant, metamer = group_name[0], group_name[1]
            if metamer == 3 or True not in data['leaf_is_emerged'].unique():
                continue
            leaf_emergence_t = data[data['leaf_is_emerged'] == True].iloc[0]['t']
            simulation_leaf_emergence[(plant, metamer)] = leaf_emergence_t

        simulation_phyllochron = {'plant': [], 'metamer': [], 'phyllochron': []}
        for key, leaf_emergence_t in sorted(simulation_leaf_emergence.items()):
            plant, metamer = key[0], key[1]
            if metamer == 4:
                continue
            simulation_phyllochron['plant'].append(plant)
            simulation_phyllochron['metamer'].append(metamer)
            prev_leaf_emergence_t = simulation_leaf_emergence[(plant, metamer - 1)]
            if simulation_df_SAM[(simulation_df_SAM['t'] == leaf_emergence_t) | (simulation_df_SAM['t'] == prev_leaf_emergence_t)].sum_TT.count() == 2:
                phyllo_DD = simulation_df_SAM[(simulation_df_SAM['t'] == leaf_emergence_t)].sum_TT.values[0] - simulation_df_SAM[(simulation_df_SAM['t'] == prev_leaf_emergence_t)].sum_TT.values[0]
            else:
                phyllo_DD = np.nan
            simulation_phyllochron['phyllochron'].append(phyllo_DD)

        # Reference simulation
        reference_df_SAM = pd.read_csv(os.path.join(reference_path, OUTPUTS_DIRPATH, AXES_OUTPUTS_FILENAME))
        reference_df_SAM = reference_df_SAM[reference_df_SAM['axis'] == 'MS']
        reference_df_hz = pd.read_csv(os.path.join(reference_path, POSTPROCESSING_DIRPATH, HIDDENZONES_POSTPROCESSING_FILENAME))
        reference_grouped_df = reference_df_hz[reference_df_hz['axis'] == 'MS'].groupby(['plant', 'metamer'])[['t', 'leaf_is_emerged']]
        reference_leaf_emergence = {}

        for group_name, data in reference_grouped_df:
            plant, metamer = group_name[0], group_name[1]
            if metamer == 3 or True not in data['leaf_is_emerged'].unique():
                continue
            leaf_emergence_t = data[data['leaf_is_emerged'] == True].iloc[0]['t']
            reference_leaf_emergence[(plant, metamer)] = leaf_emergence_t

        reference_phyllochron = {'plant': [], 'metamer': [], 'phyllochron': []}
        for key, leaf_emergence_t in sorted(reference_leaf_emergence.items()):
            plant, metamer = key[0], key[1]
            if metamer == 4:
                continue
            reference_phyllochron['plant'].append(plant)
            reference_phyllochron['metamer'].append(metamer)
            prev_leaf_emergence_t = reference_leaf_emergence[(plant, metamer - 1)]
            if reference_df_SAM[(reference_df_SAM['t'] == leaf_emergence_t) | (reference_df_SAM['t'] == prev_leaf_emergence_t)].sum_TT.count() == 2:
                phyllo_DD = reference_df_SAM[(reference_df_SAM['t'] == leaf_emergence_t)].sum_TT.values[0] - reference_df_SAM[(reference_df_SAM['t'] == prev_leaf_emergence_t)].sum_TT.values[0]
            else:
                phyllo_DD = np.nan
            reference_phyllochron['phyllochron'].append(phyllo_DD)

        if len(simulation_phyllochron['metamer']) > 0:
            fig, ax = plt.subplots()
            plt.xlim((int(min(simulation_phyllochron['metamer']) - 1), int(max(simulation_phyllochron['metamer']) + 1)))
            plt.ylim(ymin=0, ymax=150)

            ax.plot(simulation_phyllochron['metamer'], simulation_phyllochron['phyllochron'], color='b', marker='o', label='Current simulation')
            for i, j in zip(simulation_phyllochron['metamer'], simulation_phyllochron['phyllochron']):
                ax.annotate(str(int(round(j, 0))), xy=(i, j + 2), ha='center')

            ax.plot(reference_phyllochron['metamer'], reference_phyllochron['phyllochron'], color='r', marker='o', label='Reference simulation')
            for i, j in zip(reference_phyllochron['metamer'], reference_phyllochron['phyllochron']):
                ax.annotate(str(int(round(j, 0))), xy=(i, j + 2), ha='center')

            ax.set_xlabel('Leaf number')
            ax.set_ylabel('Phyllochron (Degree Day)')
            ax.set_title('phyllochron')
            plt.legend()
            plt.savefig(os.path.join(GRAPHS_DIRPATH, 'phyllochron' + '.PNG'))
            pdf_pages.savefig(fig)
            plt.close()

        # -- Phloem concentration
        # Current simulation
        simulation_organs_df = pd.read_csv(os.path.join(simulation_data_path, POSTPROCESSING_DIRPATH, ORGANS_POSTPROCESSING_FILENAME))
        simulation_phloem_df = simulation_organs_df[(simulation_organs_df['axis'] == 'MS') & (simulation_organs_df['organ'] == 'phloem')]

        # Reference simulation
        reference_organs_df = pd.read_csv(os.path.join(reference_path, POSTPROCESSING_DIRPATH, ORGANS_POSTPROCESSING_FILENAME))
        reference_phloem_df = reference_organs_df[(reference_organs_df['axis'] == 'MS') & (reference_organs_df['organ'] == 'phloem')]

        # Graphs
        vars_name = ['Conc_Sucrose', 'Conc_Amino_Acids']
        for var in vars_name:
            fig, ax = plt.subplots()
            ax.plot(simulation_phloem_df.t, simulation_phloem_df[var], color='c', label='Current simulation')
            ax.plot(reference_phloem_df.t, reference_phloem_df[var], color='r', label='Reference simulation')
            ax.set_xlabel('Time (hour)')
            ax.set_ylabel(var + '(µmol g-1 mstruct)')
            plt.legend()
            plt.savefig(os.path.join(GRAPHS_DIRPATH, var + '_phloem.PNG'))
            pdf_pages.savefig(fig)
            plt.close()

        # -- Shoot root drymass
        # Current simulation
        simulation_axes_df = pd.read_csv(os.path.join(simulation_data_path, POSTPROCESSING_DIRPATH, AXES_POSTPROCESSING_FILENAME))
        simulation_axis_df = simulation_axes_df[simulation_axes_df['axis'] == 'MS']

        # Reference simulation
        reference_axes_df = pd.read_csv(os.path.join(reference_path, POSTPROCESSING_DIRPATH, AXES_POSTPROCESSING_FILENAME))
        reference_axis_df = reference_axes_df[reference_organs_df['axis'] == 'MS']

        # Graphs
        vars_name = {'shoot_roots_ratio': 'Shoot/Roots dry mass ratio', 'C_N_ratio': 'C/N mass ratio', 'sum_dry_mass_shoot': 'sum dry mass shoot (g)',
                     'sum_dry_mass_roots': 'sum dry mass roots (g)', 'N_content_shoot': 'N franction (% dry mass)',
                     'Total_Photosynthesis': 'Cumulative photosynthesis (µmol)'}

        for var, leg in vars_name.items():
            fig, ax = plt.subplots()
            if var == 'Total_Photosynthesis':
                ax.plot(simulation_axis_df.t, simulation_axis_df[var].cumsum(), color='c', label='Current simulation')
                ax.plot(reference_axis_df.t, reference_axis_df[var].cumsum(), color='r', label='Reference simulation')
            else:
                ax.plot(simulation_axis_df.t, simulation_axis_df[var], color='c', label='Current simulation')
                ax.plot(reference_axis_df.t, reference_axis_df[var], color='r', label='Reference simulation')
            ax.set_xlabel('Time (hour)')
            ax.set_ylabel(leg)
            plt.legend()
            plt.savefig(os.path.join(GRAPHS_DIRPATH, var + '.PNG'))
            pdf_pages.savefig(fig)
            plt.close()

        # -- LAI
        # Current simulation
        simulation_df_elt = pd.read_csv(os.path.join(simulation_data_path, POSTPROCESSING_DIRPATH, ELEMENTS_POSTPROCESSING_FILENAME))
        simulation_df_elt['green_area_rep'] = simulation_df_elt.green_area * simulation_df_elt.nb_replications
        simulation_grouped_df = simulation_df_elt[(simulation_df_elt.axis == 'MS') & (simulation_df_elt.element == 'LeafElement1')].groupby(['t', 'plant'])

        simulation_LAI_dict = {'t': [], 'plant': [], 'LAI': []}
        for name, data in simulation_grouped_df:
            t, plant = name[0], name[1]
            simulation_LAI_dict['t'].append(t)
            simulation_LAI_dict['plant'].append(plant)
            simulation_LAI_dict['LAI'].append(data['green_area_rep'].sum() * PLANT_DENSITY[plant])

        # Reference simulation
        reference_df_elt = pd.read_csv(os.path.join(reference_path, POSTPROCESSING_DIRPATH, ELEMENTS_POSTPROCESSING_FILENAME))
        reference_df_elt['green_area_rep'] = reference_df_elt.green_area * reference_df_elt.nb_replications
        reference_grouped_df = reference_df_elt[(reference_df_elt.axis == 'MS') & (reference_df_elt.element == 'LeafElement1')].groupby(['t', 'plant'])

        reference_LAI_dict = {'t': [], 'plant': [], 'LAI': []}
        for name, data in reference_grouped_df:
            t, plant = name[0], name[1]
            reference_LAI_dict['t'].append(t)
            reference_LAI_dict['plant'].append(plant)
            reference_LAI_dict['LAI'].append(data['green_area_rep'].sum() * PLANT_DENSITY[plant])

        # Graph
        fig, ax = plt.subplots()
        ax.plot(simulation_LAI_dict['t'], simulation_LAI_dict['LAI'], color='c', label='Current simulation')
        ax.plot(reference_LAI_dict['t'], reference_LAI_dict['LAI'], color='r', label='Reference simulation')
        ax.set_xlabel('Time (hour)')
        ax.set_ylabel('LAI')
        plt.legend()
        plt.savefig(os.path.join(GRAPHS_DIRPATH, 'LAI' + '.PNG'))
        pdf_pages.savefig(fig)
        plt.close()

        # -- Root N uptake
        # Current simulation
        simulation_organs_df = pd.read_csv(os.path.join(simulation_data_path, POSTPROCESSING_DIRPATH, ORGANS_POSTPROCESSING_FILENAME))
        simulation_phloem_df = simulation_organs_df[(simulation_organs_df['axis'] == 'MS') & (simulation_organs_df['organ'] == 'roots')]

        # Reference simulation
        reference_organs_df = pd.read_csv(os.path.join(reference_path, POSTPROCESSING_DIRPATH, ORGANS_POSTPROCESSING_FILENAME))
        reference_phloem_df = reference_organs_df[(reference_organs_df['axis'] == 'MS') & (reference_organs_df['organ'] == 'roots')]

        # Graphs
        vars_name = ['Uptake_Nitrates']
        for var in vars_name:
            fig, ax = plt.subplots()
            ax.plot(simulation_phloem_df.t, simulation_phloem_df[var].cumsum(), color='c', label='Current simulation')
            ax.plot(reference_phloem_df.t, reference_phloem_df[var].cumsum(), color='r', label='Reference simulation')
            ax.set_xlabel('Time (hour)')
            ax.set_ylabel('Cumulative Nitrates uptake (µmol)')
            plt.legend()
            plt.savefig(os.path.join(GRAPHS_DIRPATH, var + '_roots.PNG'))
            pdf_pages.savefig(fig)
            plt.close()


if __name__ == '__main__':
    test_run(overwrite_desired_data=False)

    data_for_validation_dir = 'external soil model'
    compare_to_vegetative_stages(data_for_validation_dir, {1: 250})

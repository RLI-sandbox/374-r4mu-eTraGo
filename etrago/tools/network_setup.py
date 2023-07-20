from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import logging
import os.path

from pypsa.components import Network
from pypsa.networkclustering import aggregategenerators
import geopandas as gpd
import numpy as np
import pandas as pd

from etrago.cluster.spatial import strategies_generators
from etrago.config.config import settings
from etrago.tools.io import engine

if "READTHEDOCS" not in os.environ:
    # Sphinx does not run this code.
    # Do not import internal packages directly

    from etrago import Etrago

logger = logging.getLogger(__name__)


def setup_etrago(args: dict, json_path: str | Path | None) -> Etrago:
    etrago = Etrago(args, json_path=json_path)

    # import network from database
    etrago.build_network_from_db()

    # adjust network regarding eTraGo setting
    etrago.adjust_network()

    network = etrago.network

    # FIXME: failing consistency check
    network.lines["s_nom_opt"] = np.nan

    # clean up data types
    for col in network.lines.columns:
        if col in ["bus0", "bus1", "type", "carrier"]:
            network.lines[col] = network.lines[col].astype(str)
        elif str(network.lines[col].dtype) != "bool":
            network.lines[col] = pd.to_numeric(
                network.lines[col], errors="ignore"
            )

    network.transformers.lifetime = network.transformers.lifetime.astype(float)
    network.lines.lifetime = network.lines.lifetime.astype(float)

    # FIXME: @CB fix des aktuell fehlerhaften Szenarios
    network.mremove(
        "Generator",
        network.generators[
            ~network.generators.bus.isin(network.buses.index)
        ].index,
    )

    network.links = network.links[network.links.bus1.isin(network.buses.index)]

    network.links = network.links[network.links.bus0.isin(network.buses.index)]

    network.lines = network.lines[network.lines.bus1.isin(network.buses.index)]

    network.lines = network.lines[network.lines.bus0.isin(network.buses.index)]

    # remove none AC and DC network components
    carrier = ["DC", "AC"]

    import_gen_from_links(network)

    # drop buses
    none_ac_buses = network.buses.loc[
        ~network.buses.carrier.isin(carrier)
    ].index
    network.buses.drop(index=none_ac_buses, inplace=True)

    # drop generators
    gens_to_drop = network.generators.loc[
        network.generators.bus.isin(none_ac_buses)
    ].index
    network.generators.drop(index=gens_to_drop, inplace=True)

    # drop lines
    network.lines = network.lines.loc[network.lines.carrier.isin(carrier)]

    # drop links
    network.links = network.links.loc[network.links.carrier.isin(carrier)]

    # drop loads
    network.loads = network.loads.loc[network.loads.carrier.isin(carrier)]

    # drop storage units
    storage_units_to_drop = network.storage_units.loc[
        network.storage_units.bus.isin(none_ac_buses)
    ].index
    network.storage_units.drop(index=storage_units_to_drop, inplace=True)

    # drop stores
    stores_to_drop = network.stores.loc[
        network.stores.bus.isin(none_ac_buses)
    ].index
    network.stores.drop(index=stores_to_drop, inplace=True)

    return etrago


def import_gen_from_links(network):
    """
    Creates gas generators from links in order to not lose them when
    dropping non-electric carriers.

    Parameters
    ----------
    network : pypsa.Network object
        Container for all network components.

    Returns
    -------
    None.

    """

    # Discard all generators < 1kW
    discard_gen = network.links[network.links["p_nom"] <= 0.001].index
    network.links.drop(discard_gen, inplace=True)
    for df in network.links_t:
        if not network.links_t[df].empty:
            network.links_t[df].drop(
                columns=discard_gen.values, inplace=True, errors="ignore"
            )

    gas_to_add = network.links[
        network.links.carrier.isin(
            [
                "central_gas_CHP",
                "OCGT",
                "H2_to_power",
                "industrial_gas_CHP",
            ]
        )
    ].copy()

    # Drop generators from the links table
    network.links.drop(gas_to_add.index, inplace=True)

    gas_to_add.rename(columns={"bus1": "bus"}, inplace=True)

    # Create generators' names like in network.generators
    gas_to_add["Generator"] = (
        gas_to_add["bus"] + " " + gas_to_add.index + gas_to_add["carrier"]
    )
    gas_to_add_orig = gas_to_add.copy()
    gas_to_add.set_index("Generator", drop=True, inplace=True)
    gas_to_add = gas_to_add[
        gas_to_add.columns[gas_to_add.columns.isin(network.generators.columns)]
    ]

    network.import_components_from_dataframe(gas_to_add, "Generator")

    # Dealing with generators_t
    columns_new = network.links_t.p1.columns[
        network.links_t.p1.columns.isin(gas_to_add_orig.index)
    ]

    new_gen_t = network.links_t.p1[columns_new] * -1
    new_gen_t.rename(columns=gas_to_add_orig["Generator"], inplace=True)
    network.generators_t.p = network.generators_t.p.join(new_gen_t)

    # Drop generators from the links_t table
    for df in network.links_t:
        if not network.links_t[df].empty:
            network.links_t[df].drop(
                columns=gas_to_add_orig.index,
                inplace=True,
                errors="ignore",
            )

    # Group generators per bus if needed
    if not (
        network.generators.groupby(["bus", "carrier"]).p_nom.count() == 1
    ).all():
        network.generators["weight"] = network.generators.p_nom
        df, df_t = aggregategenerators(
            network,
            busmap=pd.Series(
                index=network.buses.index, data=network.buses.index
            ),
            custom_strategies=strategies_generators(),
        )

        # Keep control arguments from generators
        control = network.generators.groupby(
            ["bus", "carrier"]
        ).control.first()
        control.index = (
            control.index.get_level_values(0)
            + " "
            + control.index.get_level_values(1)
        )
        df.control = control

        # Drop non-aggregated generators
        network.mremove("Generator", network.generators.index)

        # Insert aggregated generators and time series
        network.import_components_from_dataframe(df, "Generator")

        for attr, data in df_t.items():
            if not data.empty:
                network.import_series_from_dataframe(data, "Generator", attr)

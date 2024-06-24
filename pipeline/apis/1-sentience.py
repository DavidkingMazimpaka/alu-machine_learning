#!/usr/bin/env python3
"""Get home planets of sentient species using SWAPI."""


import requests


def sentientPlanets():
    """
    Get names of home planets for all sentient species.
    Returns:
    list: Names of planets. Empty if none found.
    """
    url = "https://swapi-api.hbtn.io/api/species"
    r = requests.get(url)
    world_list = []
    while r.status_code == 200:
        for species in r.json()["results"]:
            url = species["homeworld"]
            if url is not None:
                ur = requests.get(url)
                world_list.append(ur.json()["name"])
        try:
            r = requests.get(r.json()["next"])
        except Exception:
            break
    return world_list

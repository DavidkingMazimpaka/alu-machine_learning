#!/usr/bin/env python3
"""Get home planets of sentient species using SWAPI."""


import requests


def sentientPlanets():
    """
    Get names of home planets for all sentient species.
    Returns:
    list: Names of planets. Empty if none found.
    """
    base_url = "https://swapi.dev/api/species/"
    planets = set()
    while base_url:
        response = requests.get(base_url)
        data = response.json()
        for species in data['results']:
            if species['classification'] == 'sentient':
                if species['homeworld']:
                    planet_response = requests.get(species['homeworld'])
                    planet_data = planet_response.json()
                    planets.add(planet_data['name'])
        base_url = data['next']
    return list(planets)

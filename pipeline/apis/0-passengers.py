#!/usr/bin/env python3
"""Find Star Wars ships to hold number of passengers using SWAPI."""


import requests


def availableShips(passengerCount):
    """
    Get Star Wars ships that can hold at least the specified number of passengers.
    Args:
    passengerCount (int): Minimum passenger capacity.
    Returns:
    list: Names of ships meeting the capacity requirement. Empty if none found.
    """
    base_url = "https://swapi.dev/api/starships/"
    ships = []
    while base_url:
        response = requests.get(base_url)
        data = response.json()
        
        for ship in data['results']:
            if ship['passengers'] != 'n/a' and ship['passengers'] != 'unknown':
                ship_capacity = int(ship['passengers'].replace(',', ''))
                if ship_capacity >= passengerCount:
                    ships.append(ship['name'])
        
        base_url = data['next']
    return ships
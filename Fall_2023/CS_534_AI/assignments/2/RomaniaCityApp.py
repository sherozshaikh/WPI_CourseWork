import sys
from SimpleProblemSolvingAgent import SimpleProblemSolvingAgent_SPSAP,romania_map

def main() -> None:
    """
    This is the main function of the Romania Map Path Finder script.

    This script allows users to find the best path between two cities in Romania
    using a Simple Problem Solving Agent (SPSA).

    Usage:
    1. Display a list of all possible Romanian cities that can be traveled to.
    2. Find the best path between two selected cities using 4 different Algorithms by providing the algorithm name, path, and cost incurred.

    Returns:
    None

    """
    all_nodes : list = sorted(romania_map.nodes())
    print('\nHere are all the possible Romania cities that can be traveled: ')
    print(all_nodes)
    spsa = SimpleProblemSolvingAgent_SPSAP()
    get_origin_city, get_destination_city = spsa._get_a_b_cities(all_nodes)
    spsa.run_models(get_origin_city,get_destination_city)
    re_run_app_condition : bool = True
    while re_run_app_condition:
        re_run_app : str = input('\nWould you like to find the best path between other two cities? ')
        if 'n' in re_run_app[0].lower():
            print('\nThank You for Using Our App')
            re_run_app_condition : bool = False
        else:
            get_origin_city, get_destination_city = spsa._get_a_b_cities(all_nodes)
            spsa.run_models(get_origin_city,get_destination_city)
    return None
if __name__ == '__main__':
    main()
    sys.exit()

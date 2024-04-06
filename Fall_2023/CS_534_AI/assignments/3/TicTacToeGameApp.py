import sys
import TicTacToeClass

def main() -> None:
    """Main function to run the Tic-Tac-Toe game simulation."""
    ttt_obj = TicTacToeClass.TicTacToeSimulation()
    ttt_obj.game_simulation()
    re_run_app_condition: bool = True
    while re_run_app_condition:
        re_run_app: str = input('\nWould you like to play the game again? ')
        re_run_app: str = re_run_app[0].lower()
        if 'n' in re_run_app:
            print('\nThank You for Playing Our Game!')
            re_run_app_condition: bool = False
        elif 'y' in re_run_app:
            ttt_obj.game_simulation()
        else:
            print(
                '\nIncorrect Entry. Please Select "Y / Yes / y / yes" for Playing and "N / No / n / no" for quitting!')
    return None


if __name__ == '__main__':
    main()
    sys.exit()

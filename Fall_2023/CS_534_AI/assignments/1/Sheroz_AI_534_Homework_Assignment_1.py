
def main() -> None:

    '''
    Takes input from the user and prints out the keyword
    Hello along with the text entered by user.
    Whatever the user enters will be assumed to be string.
    '''

    stxt: str = input("Enter your name: ")
    print(f'Hello {stxt}')
    return None


if __name__ == '__main__':

    '''
    __main__ is used whenever we aim to run
    the commands mentioned after this line.
    The commands runs only when this specific is .py file
    is called independently.
    The below commands won't work when this
    .py file is imported in some other script.
    Great for debugging the given functionality before we
    import this script in some other script.
    '''

    main()

    '''
    I remove the python objects and methods after use to
    save memory as it comes handy when we have multiple functions
    that we need to run sequentially and we run out of memory.
    '''

    del main


if __name__ == '__main__':
    import MyPhotometry
    import traceback
    try:
        MyPhotometry.run_app()
    except:
        print('\n************\n\n')
        traceback.print_exc()
        x = input('Press enter to exit.\n')
        raise

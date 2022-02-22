import computation
def main():
    """Run the simulation. It's equivalent to Light_BEC.m"""


    nj = 10
    stepJ = 1000  # counts of updating energy constraint
    computation.compute_BEC(nj, stepJ, 0) 

    return

if __name__ == "__main__":
    main()
import random
def main():
    """
    setup: 
    n is length of square
    thinning is prob of object at a point
    pchars is possible chars
    """
    n = 30
    thinning = .36
    pchars = [chr(c) for c in range(ord("A"), ord("Z")+1)]

    # TODO: why? remove P! (apparently)
    pchars.remove("P")

    # now create the ascii box, ensure proper padding with asterisks and spaces
    result = [["*" for _ in range(n)]]
    for _ in range(n-2):
        subresult = [" "]
        lenn = 1
        while lenn < n-3:
            if random.random() < thinning:
                subresult.append(random.choice(pchars))
                lenn += 1
            subresult.append(" ")
            lenn += 1
        while lenn < n-2:
            subresult.append(" ")
            lenn += 1
        result.append(["*"] + subresult + ["*"])
    result.append( ["*" for _ in range(n)])

    print(f"n = {n}, object density = {thinning}")
    # print out result
    for r in result:
        madestr = "".join(r)
        print(madestr)

# 
main()
class iter():

    def _(arr):
        final = []

        for i in range(len(arr)):

            final.append(i)
            for x in range(len(arr[i])):
                final[i].append(x)

        return final
                
#include <stdio.h>
#include <stdlib.h>

void BinaryToText(char const *inputFile, char const *outputFile) {
    size_t num;
    char str[64];

    FILE *finp = fopen(inputFile, "rb");
    FILE *fout = fopen(outputFile, "w");

    while (!feof(finp)) {
        fread(&num, sizeof(size_t), 1, finp);
        sprintf(str, "%ld", num)
        fprintf(fout, "%s\n", (char*)str);
    }
    fclose(finp);
    fclose(fout);
}

int main(int argc, char const *argv[])
{
    BinaryToText(argv[1], argv[2]);
    return 0;
}
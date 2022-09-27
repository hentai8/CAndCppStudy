#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <atomic>


struct structB {
    int b;
    ssize_t c;
};

struct structA {
    int a;
    std::atomic<uint32_t> ref_count;
    structB b;
};
void print_structA(struct structA *structa);


int main()
{
    printf("000");
    struct structA structa;
    structa.a = 123;
    structa.b.b = 999;
    structa.b.c = 123;
    print_structA(&structa);
    printf("321");
    // print_structA()
    return 0;
}

void print_structA(struct structA *struct0){
    printf("Book title : %d\n",struct0->a);
    printf("Book title : %d\n",struct0->b.b);
    printf("sszie_t : %zd\n",struct0->b.c);
    printf("ref_count: %u\n",struct0->ref_count.load());
}
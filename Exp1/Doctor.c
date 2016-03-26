#include <stdio.h>
#include <string.h>

int main() {
    int n; //事件数
    int queue[3][2000]; //队列病人的优先级
    int ids[3][2000]; //队列病人的id
    int id; 
    int i, j, k;
    char type[5]; //事件类型
    int length[3]; //队列长度
    int prior; //优先级
    int doctor;
    int l;
    while (scanf("%d", &n) != EOF) {
        id = 0;
        for (i = 0; i < 3; ++i) length[3] = 0;
        for (i = 0; i < n; ++i) {
            scanf("%s", type);
            if (strcmp(type, "IN") == 0) {
                scanf("%d", &doctor);
                scanf("%d", &prior);
                l = length[doctor - 1]++;
                queue[doctor - 1][l] = prior;
                ids[doctor - 1][l] = id++;
            }
            if (strcmp(type, "OUT") == 0) {
                scanf("%d", &doctor);
                int max = -1, maxId = -1, maxn = 0;
                //找出最高优先级的病人
                for (j = 0; j < length[doctor - 1]; ++j) {
                    if (queue[doctor - 1][j] > max) {
                        max = queue[doctor - 1][j];
                        maxId = ids[doctor - 1][j];
                        maxn = j;
                    }
                }
                if (max != -1) {
                    printf("%d\n", maxId + 1); queue[doctor - 1][maxn] = -1;
                }
                else printf("EMPTY\n");
            }

        }
    }
    return 0;
}




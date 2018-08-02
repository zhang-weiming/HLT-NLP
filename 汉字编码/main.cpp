#include <stdio.h>

int countOneInLeft(char a) {
	unsigned char b = 0x80;
	int c = 0;
	for (int i = 0; (a & b) != 0 && i < 8; i++) {
		c++;
		b = b >> 1;
	}
	return c;
}

/*
UTF-8编码规则：
如果只有一个字节则其最高二进制位为0；
如果是多字节，其第一个字节从最高位开始，连续的二进制位值为1的个数决定了其编码的位数，其余各字节均以10开头。
UTF-8最多可用到6个字节。
如表：
1字节 0xxxxxxx
2字节 110xxxxx 10xxxxxx
3字节 1110xxxx 10xxxxxx 10xxxxxx
4字节 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
5字节 111110xx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
6字节 1111110x 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
*/
void readWithUTF8(const char* filename) {
	FILE* fp = fopen(filename, "r");
	if (fp == NULL) {
		printf("无法打开文件%s！\n", filename);
		return;
	}

	char ch = fgetc(fp);
	char tmp[9] = {0};
	int cnt = 0; 
	while (ch != EOF) {
		cnt++;
		int n = countOneInLeft(ch);
		if (n == 0) {
			printf("%c", ch);
			ch = fgetc(fp);
		}
		else if (n < 8) {
			for (int i = 0; i < n; i++) {
				tmp[i] = ch;
				ch = fgetc(fp);
			}
			printf("%s", tmp); 
		}
		else {
			printf("?");
			ch = fgetc(fp);
		}
		printf(" ");
//		switch (countOneInLeft(ch)) {
//			case 0:
//				printf("%c", ch);
//				break;
//			case 2:
//				tmp[0] = ch;
//				tmp[1] = fgetc(fp);
//				printf("%c%c", tmp[0], tmp[1]);
//				break;
//			case 3:
//				tmp[0] = ch;
//				tmp[1] = fgetc(fp);
//				tmp[2] = fgetc(fp);
//				printf("%c%c%c", tmp[0], tmp[1], tmp[2]);
//				break;
//			default:
//				printf("?");
//				return;
//		}
//		printf(" ");
//		ch = fgetc(fp);
	}
	printf("%d\n", cnt);
	fclose(fp);
}

/*
如果只有一个字节则其最高二进制位为0；
否则，一定是由两个字节组成。
*/
void readWithGBK(const char* filename) {
	FILE* fp = fopen(filename, "r");
	if (fp == NULL) {
		printf("无法打开文件%s！\n", filename);
		return;
	}

	char ch = fgetc(fp);
	int cnt = 0;
	while (ch != EOF) {
		cnt++;
		if (ch < 0) {
			printf("%c%c", ch, fgetc(fp));
		}
		else {
			printf("%c", ch);
		}
		printf(" ");
		ch = fgetc(fp);
	}
	printf("%d\n", cnt);
	fclose(fp);
}

int main() {
	readWithUTF8("./data/data-utf8.txt");
	readWithGBK("./data/data-gbk.txt");

	return 0;
}

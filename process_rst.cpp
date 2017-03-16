#include <iostream>
#include <fstream>
#include <string.h>

using namespace std;

int main(int argc, char *argv[]) {

	ifstream ifile;
	ofstream ofile;
	
	ifile.open(argv[1]);
	ofile.open(argv[2]);

	char s[10];
	char graphname[100];
	float time[7];
	float t;
	int i = 0;

	string str;

	while (getline(ifile, str)) {
		if (strstr(str.c_str(), "./scc")) {
			sscanf(str.c_str(), "%s%s%s%s%s%s", graphname, graphname, graphname, graphname, graphname, graphname);
			ofile << graphname << ' ';
		}
		if (strstr(str.c_str(), "time:")) {
			//cout << str;
			sscanf(str.c_str(), "%s%f", s, &t);
			time[(i++) % 7] = t;
			if (i % 7 == 0) {
				ofile << time[0] << ' ' << time[1] << ' ' << time[2] << ' ' << time[3] << ' ' << time[4] << ' ' << time[5] << ' ' << time[6] << ' ' << endl;
			}
		}
	}
	ifile.close();
	ofile.close();

	return 0;
}

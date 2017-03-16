#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "load.h"


#include <iostream>
#include <fstream>
#include <string>
#include <string.h>
#include <vector>
#include <set>

using namespace std;

//bool load(char *file, bool header, unsigned *countOfNodes, unsigned *countOfEdges, unsigned **nodes_addr, unsigned **edges_addr, unsigned **nodesT_addr, unsigned **edgesT_addr) {
bool load(char *file, bool header, unsigned *countOfNodes, unsigned *countOfEdges, unsigned **nodes_addr, unsigned **edges_addr, unsigned **nodesT_addr, unsigned **edgesT_addr, unsigned *outDegree, unsigned *inDegree) {

	std::ifstream cfile;
	cfile.open(file);

	bool pattern = false;
	bool symmetric = false;
	float tmp;

	std::string str;
	getline(cfile, str);

	if (strstr(str.c_str(), "pattern"))
		pattern = true;
	if (strstr(str.c_str(), "symmetric"))
		symmetric = true;

	printf("%d %d\n", pattern?1:0, symmetric?1:0);

	char c;
	sscanf(str.c_str(), "%c", &c);

	while (c == '%') {
		getline(cfile, str);
		sscanf(str.c_str(), "%c", &c);
	}

	int m, n, nnz;
	if (strstr(file, ".mtx")) {
		sscanf(str.c_str(), "%d %d %d", &m, &n, &nnz);
	printf("m=%d\tn=%d\tnnz=%d\n", m, n, nnz);
	if (m != n) {
		printf("error!\n");
		exit(1);
	}
	}
	else
		sscanf(str.c_str(), "%d %d", &m, &nnz);

	*countOfNodes = m;
	*countOfEdges = nnz;	

	int max_in, min_in, max_out, min_out;
	max_in = max_out = 0;
	min_in = min_out = m;
	float average;
	average = nnz * 1.0 / m;

	vector<set<int> > svector;
	vector<set<int> > svectorT;
	set<int> s;
	for (int i = 0; i < *countOfNodes + 1; i++) {
		svector.push_back(s);
		if (!symmetric)
			svectorT.push_back(s);
	}

	int dst, src;
	for (int i = 0; i < nnz; i++) {
		getline(cfile, str);
		if (pattern)
			sscanf(str.c_str(), "%d %d", &dst, &src);
		else
			sscanf(str.c_str(), "%d %d %f", &dst, &src, &tmp);
#if 1
		if (symmetric) {
			if (dst != src) {
				(*countOfEdges)++;
				svector[dst].insert(src);
			}
			svector[src].insert(dst);
		}
		else {
			svector[src].insert(dst);
			svectorT[dst].insert(src);
		}
#endif

#if 0
		if (symmetric) {
			if (dst == src) {
				(*countOfEdges)--;
			}
			else {
				(*countOfEdges)++;
				svector[dst].insert(src);
				svector[src].insert(dst);
			}
		}
		else {
			if (dst == src) {
				(*countOfEdges)--;
			}
			else {
				svector[src].insert(dst);
				svectorT[dst].insert(src);
			}
		}
#endif

	}
	cfile.close();

	printf("nodes=%d\tedges=%d\n", *countOfNodes, *countOfEdges);

	printf("OK\n");

	unsigned *nodes = NULL;
	unsigned *edges = NULL;
	unsigned *nodesT = NULL;
	unsigned *edgesT = NULL;

	nodes = (unsigned *)malloc((*countOfNodes + 2) * sizeof(unsigned));
	edges = (unsigned *)malloc((*countOfEdges) * sizeof(unsigned));

	outDegree = (unsigned *)malloc((*countOfNodes + 2) * sizeof(unsigned));
	inDegree = (unsigned *)malloc((*countOfNodes + 2) * sizeof(unsigned));
	

	if (!symmetric) {
		nodesT = (unsigned *)malloc((*countOfNodes + 2) * sizeof(unsigned));
		edgesT = (unsigned *)malloc((*countOfEdges) * sizeof(unsigned));
	}

	int count = 0;
	int degree;
	for (int i = 1; i < *countOfNodes + 1; i++) {
		nodes[i] = count;
		degree = svector[i].size();
		outDegree[i] = degree;
		count += degree;

		if (degree > max_out)
			max_out = degree;
		if (degree < min_out)
			min_out = degree;
	}
	nodes[*countOfNodes + 1] = count;

	set<int>::iterator site;
	for (int i = 1, index = 0; i < *countOfNodes + 1; i++) {
		site = svector[i].begin();
		while (site != svector[i].end()) {
			edges[index++] = *site;
			site++;
		}
	}

	printf("OK\n");
	
	if (!symmetric) {
		int count = 0;
		for (int i = 1; i < *countOfNodes + 1; i++) {
			nodesT[i] = count;
			degree = svectorT[i].size();
			inDegree[i] = degree;
			count += degree;
			
			if (degree > max_in)
				max_in = degree;
			if (degree < min_in)
				min_in = degree;
		}
		nodesT[*countOfNodes + 1] = count;

		set<int>::iterator site;
		for (int i = 1, index = 0; i < *countOfNodes + 1; i++) {
			site = svectorT[i].begin();
			while (site != svectorT[i].end()) {
				edgesT[index++] = *site;
				site++;
			}
		}
	}	

	*nodes_addr = nodes;
	*edges_addr = edges;

	*nodesT_addr = nodesT;
	*edgesT_addr = edgesT;

	if (symmetric) {
		*nodesT_addr = nodes;
		*edgesT_addr = edges;
	}

	printf("max_in=%d min_in=%d max_out=%d min_out=%d average=%f\n", max_in, min_in, max_out, min_out, average);

	return true;
}

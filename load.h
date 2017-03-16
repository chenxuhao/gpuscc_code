#ifndef LOAD_H
#define LOAD_H

//bool load(char *file, bool header, unsigned *countOfNodes, unsigned *countOfEdges, unsigned **nodes, unsigned **edges, unsigned **nodesT, unsigned **edgesT);
bool load(char *file, bool header, unsigned *countOfNodes, unsigned *countOfEdges, unsigned **nodes, unsigned **edges, unsigned **nodesT, unsigned **edgesT, unsigned *outDegree, unsigned *inDegree);

#endif

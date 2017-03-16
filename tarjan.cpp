#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stack>
#include <utility>

#include "timing.h"
#include "tarjan.h"

#include <string.h>
#include <set>
#include <vector>

#define BIT_SHIFT ((unsigned)1 << 31)

//#define DEBUG

using namespace std;

class Edge
{
//private:
public:
    unsigned value;

    Edge() : value(0) {}
    Edge( unsigned & val ) : value(val) {}
    inline unsigned getValue() { return (value & 0x7FFFFFFF); }
    inline void setValue(unsigned n) { value = (n | (value & BIT_SHIFT)); }
    inline bool isValid() { return (value & BIT_SHIFT); }
    inline void setValidBit() { value = (value | BIT_SHIFT); };
    inline void clearValidBit() { value = (value & ~BIT_SHIFT); };
};

class TR_vertex
{
//private:
public:
	unsigned visit;
	unsigned low_link;

	TR_vertex() : visit(0), low_link(0) {}
	inline unsigned getVisited() { return visit; }
	inline void setVisited(unsigned n) { visit = n; }

	inline unsigned getLowLink() { return (low_link & 0x7FFFFFFF); }
	inline void setLowLink(unsigned n) { low_link = (n | (low_link & BIT_SHIFT)); }
	inline bool isInComponent () { return (low_link & BIT_SHIFT); }
	inline void setInComponentBit() { low_link = (low_link | BIT_SHIFT); };
	inline void clearInComponentBit() { low_link = (low_link & ~BIT_SHIFT); };
};

class TR_stack_vertex {
//private:
public:
    unsigned id;
    unsigned from;

    TR_stack_vertex() : id(0), from(0) {}
    inline unsigned getId() { return (id & 0x7FFFFFFF); }
    inline void setId(unsigned n) { id = (n | (id & BIT_SHIFT)); }
    inline bool isExpanded () { return (id & BIT_SHIFT); }
    inline void setExpandedBit() { id = (id | BIT_SHIFT); };
    inline void clearExpandedBit() { id = (id & ~BIT_SHIFT); };

    inline unsigned getFrom() { return from; }
    inline void setFrom(unsigned n) { from = n ; }

};


pair <unsigned, float> TR_Decomposition(unsigned CSize, unsigned RSize, unsigned *Fc, unsigned *Fr) //CSize - number of edges, RSize - number of nodes, *Fc - edges, *Fr - nodes
{
	//for (int i = 0; i < RSize+1; i ++) printf("r[%d]=%d, ", i, Fr[i]); printf("\n");
	//for (int i = 0; i < CSize; i ++) printf("c[%d]=%d, ", i, Fc[i]);  printf("\n");
	unsigned number_of_SCC = 0;
	bool terminated = false;

	unsigned trivial = 0;
	unsigned non_trivial = 0;
#ifdef DEBUG
	FILE *fp_trivial;
	fp_trivial = fopen("tarjan.trivial", "w");
	set<int> trivial_set;
	FILE *fp_dist;
	fp_dist = fopen("tarjan_dist.txt", "w");
	int *scc_dist = NULL;
	scc_dist = (int *)malloc((1 + RSize) * sizeof(int));
	memset(scc_dist, 0, (1 + RSize) * sizeof(int));
#endif

	int *scc_root;
	scc_root = (int *)malloc((RSize + 1) * sizeof(*scc_root));

	unsigned biggestSCC = 0;

	stack<TR_stack_vertex> visit_stack;
	stack<unsigned> scc_stack;

	unsigned scc_top;
	unsigned time = 1; // 0 is null vertex

	TR_stack_vertex stack_vertex;
	TR_vertex *m = new TR_vertex[ RSize + 1 ]; //originally RSize - 1, but it was wrong
	
	timeval timer;

	startTimer(&timer);
	//first initial states
	stack_vertex.setId(1);
	visit_stack.push(stack_vertex);
	unsigned i = 1;
	do {
		while ( !(visit_stack.empty()) ) {
			stack_vertex = visit_stack.top();
			visit_stack.pop();
			if ( ! stack_vertex.isExpanded() ) {
				if (m[stack_vertex.getId()].getVisited() == 0) {//states hasn't been visited during DFS yet
					m[stack_vertex.getId()].setVisited(time);
					m[stack_vertex.getId()].setLowLink(time);
					time++;
					scc_stack.push(stack_vertex.getId());
					stack_vertex.setExpandedBit();
					visit_stack.push(stack_vertex);
					for ( unsigned column = Fr[ stack_vertex.getId() ]; column < Fr[ stack_vertex.getId() + 1 ]; column++ ) {
						TR_stack_vertex succ_stack_vertex;
						succ_stack_vertex.setId(Fc[ column ]);
						succ_stack_vertex.setFrom(stack_vertex.getId());
						visit_stack.push(succ_stack_vertex);
					}
				}
				else {
					if ( ! m[stack_vertex.getId()].isInComponent() ) {
						if ( m[ stack_vertex.getFrom() ].getLowLink() > m[stack_vertex.getId()].getVisited() ) {
							m[ stack_vertex.getFrom() ].setLowLink(m[stack_vertex.getId()].getVisited());
						}
					}
				}
			}
			else {
				if ( ( m[ stack_vertex.getId() ].getVisited() == m[ stack_vertex.getId() ].getLowLink() ) &&
					( ! m[ stack_vertex.getId() ].isInComponent() ) ) {
					unsigned size_of_SCC = 0;
					do {
						scc_top = scc_stack.top();

						scc_root[scc_top] = stack_vertex.getId();

						scc_stack.pop();
						m[ scc_top ].setInComponentBit();
						size_of_SCC++;
					} while ( scc_top != stack_vertex.getId() );
					number_of_SCC++;

					if (size_of_SCC > biggestSCC)
						biggestSCC = size_of_SCC;

					if (size_of_SCC == 1) {
						trivial++;
						#ifdef DEBUG
						//fprintf(fp_trivial, "%d\n", scc_top);
						trivial_set.insert(scc_top);
						#endif
					}
					else {
						non_trivial++;
					}
				}// second condition due to initial states
				if ( ( ! m[ stack_vertex.getId() ].isInComponent() ) && ( stack_vertex.getFrom() != 0 ) ) {
					if ( m[ stack_vertex.getFrom() ].getLowLink() > m[ stack_vertex.getId() ].getLowLink() ) {
						m[ stack_vertex.getFrom() ].setLowLink( m[ stack_vertex.getId() ].getLowLink());
					}
				}
			}
		}

		terminated = true;

		for (; i <= RSize; i++ ) { //orginally i < RSize - 1, but it was wrong
			if ( m[i].getVisited() == 0 ) {
				terminated = false;
				TR_stack_vertex stack_vertex;
				stack_vertex.setId(i);
				visit_stack.push(stack_vertex); 
				//cout <<" Seed = "<< i << endl;
				break;
			}
		}

	} while ( !terminated );

	float f = float(stopTimer(&timer))/1000000;
	printf("Tarjan done!\n");
#ifdef DEBUG
	set<int>::iterator ite;
	for (ite = trivial_set.begin(); ite != trivial_set.end(); ite++) {
		fprintf(fp_trivial, "%d\n", *ite);
	}
	fclose(fp_trivial);
#endif

#ifdef DEBUG
	FILE *fp_scc = fopen("tarjan_scc.txt", "w");
	int *indices;
	indices = (int *)malloc((1 + RSize) * sizeof(int));
	int count = 0;
	vector<set<int> > vs(1 + RSize);
	printf("vector init done!\n");
	for (int i = 1; i <= RSize; i++) {
		vs[scc_root[i]].insert(i);
		if (scc_root[i] == i) {
			indices[++count] = i;
		}
	}
	printf("count=%d\n", count);	
	set<int>::iterator site;
	int sum = 0;	
	for (int i = 1; i <= count; i++) {
		site = vs[indices[i]].begin();
		while (site != vs[indices[i]].end()) {
			fprintf(fp_scc, "%d ", *site);
			site++;
		}
		fprintf(fp_scc, "\n");
		sum += vs[indices[i]].size();
		scc_dist[vs[indices[i]].size()]++;
	}
	fclose(fp_scc);
	printf("sum=%d\n", sum);
	for (int i = 1; i <= RSize; i++) {
		if (scc_dist[i] != 0)
			fprintf(fp_dist, "%d %d\n", i, scc_dist[i]);
	}
	fclose(fp_dist);
#endif

	printf("trivial = %d\tnon_trivial=%d\tnSCC = %d\tbiggestSCC = %d\n", trivial, non_trivial, number_of_SCC, biggestSCC);
	
	return make_pair(number_of_SCC, f);
}

#ifdef DEBUG
#undef DEBUG 
#endif

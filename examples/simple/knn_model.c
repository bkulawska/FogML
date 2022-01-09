/*
   Copyright 2021 FogML

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

double sqrt(double number){
    double error = 0.00001;
    double s = number;

    while ((s - number / s) > error)
    {
        s = (s + number / s) / 2;
    }
    return s;
}

double pow2(double x){
	return x*x;
}

typedef struct Point{
    int idx;
    double distance;
} Point;

void swap(Point* tab, int i1, int i2){
    Point temp = tab[i1];
    tab[i1] = tab[i2];
    tab[i2] = temp;
}

void quick_sort(Point *tab, int left, int right){
    if(right <= left){
        return;
    }

    int i = left - 1;
	int j = right + 1;
    double pivot = tab[(left+right)/2].distance;

    while(1)
    {
        while(pivot>tab[++i].distance);
        while(pivot<tab[--j].distance);
        if(i <= j){
            swap(tab,i,j);
        }
        else {
            break;
        }
    }
    if(j > left){
        quick_sort(tab, left, j);
    }
    if(i < right){
        quick_sort(tab, i, right);
    }
}

int classifier(double* x){
	double attributes[150][4] = {
{5.100000, 3.500000, 1.400000, 0.200000, },
{4.900000, 3.000000, 1.400000, 0.200000, },
{4.700000, 3.200000, 1.300000, 0.200000, },
{4.600000, 3.100000, 1.500000, 0.200000, },
{5.000000, 3.600000, 1.400000, 0.200000, },
{5.400000, 3.900000, 1.700000, 0.400000, },
{4.600000, 3.400000, 1.400000, 0.300000, },
{5.000000, 3.400000, 1.500000, 0.200000, },
{4.400000, 2.900000, 1.400000, 0.200000, },
{4.900000, 3.100000, 1.500000, 0.100000, },
{5.400000, 3.700000, 1.500000, 0.200000, },
{4.800000, 3.400000, 1.600000, 0.200000, },
{4.800000, 3.000000, 1.400000, 0.100000, },
{4.300000, 3.000000, 1.100000, 0.100000, },
{5.800000, 4.000000, 1.200000, 0.200000, },
{5.700000, 4.400000, 1.500000, 0.400000, },
{5.400000, 3.900000, 1.300000, 0.400000, },
{5.100000, 3.500000, 1.400000, 0.300000, },
{5.700000, 3.800000, 1.700000, 0.300000, },
{5.100000, 3.800000, 1.500000, 0.300000, },
{5.400000, 3.400000, 1.700000, 0.200000, },
{5.100000, 3.700000, 1.500000, 0.400000, },
{4.600000, 3.600000, 1.000000, 0.200000, },
{5.100000, 3.300000, 1.700000, 0.500000, },
{4.800000, 3.400000, 1.900000, 0.200000, },
{5.000000, 3.000000, 1.600000, 0.200000, },
{5.000000, 3.400000, 1.600000, 0.400000, },
{5.200000, 3.500000, 1.500000, 0.200000, },
{5.200000, 3.400000, 1.400000, 0.200000, },
{4.700000, 3.200000, 1.600000, 0.200000, },
{4.800000, 3.100000, 1.600000, 0.200000, },
{5.400000, 3.400000, 1.500000, 0.400000, },
{5.200000, 4.100000, 1.500000, 0.100000, },
{5.500000, 4.200000, 1.400000, 0.200000, },
{4.900000, 3.100000, 1.500000, 0.200000, },
{5.000000, 3.200000, 1.200000, 0.200000, },
{5.500000, 3.500000, 1.300000, 0.200000, },
{4.900000, 3.600000, 1.400000, 0.100000, },
{4.400000, 3.000000, 1.300000, 0.200000, },
{5.100000, 3.400000, 1.500000, 0.200000, },
{5.000000, 3.500000, 1.300000, 0.300000, },
{4.500000, 2.300000, 1.300000, 0.300000, },
{4.400000, 3.200000, 1.300000, 0.200000, },
{5.000000, 3.500000, 1.600000, 0.600000, },
{5.100000, 3.800000, 1.900000, 0.400000, },
{4.800000, 3.000000, 1.400000, 0.300000, },
{5.100000, 3.800000, 1.600000, 0.200000, },
{4.600000, 3.200000, 1.400000, 0.200000, },
{5.300000, 3.700000, 1.500000, 0.200000, },
{5.000000, 3.300000, 1.400000, 0.200000, },
{7.000000, 3.200000, 4.700000, 1.400000, },
{6.400000, 3.200000, 4.500000, 1.500000, },
{6.900000, 3.100000, 4.900000, 1.500000, },
{5.500000, 2.300000, 4.000000, 1.300000, },
{6.500000, 2.800000, 4.600000, 1.500000, },
{5.700000, 2.800000, 4.500000, 1.300000, },
{6.300000, 3.300000, 4.700000, 1.600000, },
{4.900000, 2.400000, 3.300000, 1.000000, },
{6.600000, 2.900000, 4.600000, 1.300000, },
{5.200000, 2.700000, 3.900000, 1.400000, },
{5.000000, 2.000000, 3.500000, 1.000000, },
{5.900000, 3.000000, 4.200000, 1.500000, },
{6.000000, 2.200000, 4.000000, 1.000000, },
{6.100000, 2.900000, 4.700000, 1.400000, },
{5.600000, 2.900000, 3.600000, 1.300000, },
{6.700000, 3.100000, 4.400000, 1.400000, },
{5.600000, 3.000000, 4.500000, 1.500000, },
{5.800000, 2.700000, 4.100000, 1.000000, },
{6.200000, 2.200000, 4.500000, 1.500000, },
{5.600000, 2.500000, 3.900000, 1.100000, },
{5.900000, 3.200000, 4.800000, 1.800000, },
{6.100000, 2.800000, 4.000000, 1.300000, },
{6.300000, 2.500000, 4.900000, 1.500000, },
{6.100000, 2.800000, 4.700000, 1.200000, },
{6.400000, 2.900000, 4.300000, 1.300000, },
{6.600000, 3.000000, 4.400000, 1.400000, },
{6.800000, 2.800000, 4.800000, 1.400000, },
{6.700000, 3.000000, 5.000000, 1.700000, },
{6.000000, 2.900000, 4.500000, 1.500000, },
{5.700000, 2.600000, 3.500000, 1.000000, },
{5.500000, 2.400000, 3.800000, 1.100000, },
{5.500000, 2.400000, 3.700000, 1.000000, },
{5.800000, 2.700000, 3.900000, 1.200000, },
{6.000000, 2.700000, 5.100000, 1.600000, },
{5.400000, 3.000000, 4.500000, 1.500000, },
{6.000000, 3.400000, 4.500000, 1.600000, },
{6.700000, 3.100000, 4.700000, 1.500000, },
{6.300000, 2.300000, 4.400000, 1.300000, },
{5.600000, 3.000000, 4.100000, 1.300000, },
{5.500000, 2.500000, 4.000000, 1.300000, },
{5.500000, 2.600000, 4.400000, 1.200000, },
{6.100000, 3.000000, 4.600000, 1.400000, },
{5.800000, 2.600000, 4.000000, 1.200000, },
{5.000000, 2.300000, 3.300000, 1.000000, },
{5.600000, 2.700000, 4.200000, 1.300000, },
{5.700000, 3.000000, 4.200000, 1.200000, },
{5.700000, 2.900000, 4.200000, 1.300000, },
{6.200000, 2.900000, 4.300000, 1.300000, },
{5.100000, 2.500000, 3.000000, 1.100000, },
{5.700000, 2.800000, 4.100000, 1.300000, },
{6.300000, 3.300000, 6.000000, 2.500000, },
{5.800000, 2.700000, 5.100000, 1.900000, },
{7.100000, 3.000000, 5.900000, 2.100000, },
{6.300000, 2.900000, 5.600000, 1.800000, },
{6.500000, 3.000000, 5.800000, 2.200000, },
{7.600000, 3.000000, 6.600000, 2.100000, },
{4.900000, 2.500000, 4.500000, 1.700000, },
{7.300000, 2.900000, 6.300000, 1.800000, },
{6.700000, 2.500000, 5.800000, 1.800000, },
{7.200000, 3.600000, 6.100000, 2.500000, },
{6.500000, 3.200000, 5.100000, 2.000000, },
{6.400000, 2.700000, 5.300000, 1.900000, },
{6.800000, 3.000000, 5.500000, 2.100000, },
{5.700000, 2.500000, 5.000000, 2.000000, },
{5.800000, 2.800000, 5.100000, 2.400000, },
{6.400000, 3.200000, 5.300000, 2.300000, },
{6.500000, 3.000000, 5.500000, 1.800000, },
{7.700000, 3.800000, 6.700000, 2.200000, },
{7.700000, 2.600000, 6.900000, 2.300000, },
{6.000000, 2.200000, 5.000000, 1.500000, },
{6.900000, 3.200000, 5.700000, 2.300000, },
{5.600000, 2.800000, 4.900000, 2.000000, },
{7.700000, 2.800000, 6.700000, 2.000000, },
{6.300000, 2.700000, 4.900000, 1.800000, },
{6.700000, 3.300000, 5.700000, 2.100000, },
{7.200000, 3.200000, 6.000000, 1.800000, },
{6.200000, 2.800000, 4.800000, 1.800000, },
{6.100000, 3.000000, 4.900000, 1.800000, },
{6.400000, 2.800000, 5.600000, 2.100000, },
{7.200000, 3.000000, 5.800000, 1.600000, },
{7.400000, 2.800000, 6.100000, 1.900000, },
{7.900000, 3.800000, 6.400000, 2.000000, },
{6.400000, 2.800000, 5.600000, 2.200000, },
{6.300000, 2.800000, 5.100000, 1.500000, },
{6.100000, 2.600000, 5.600000, 1.400000, },
{7.700000, 3.000000, 6.100000, 2.300000, },
{6.300000, 3.400000, 5.600000, 2.400000, },
{6.400000, 3.100000, 5.500000, 1.800000, },
{6.000000, 3.000000, 4.800000, 1.800000, },
{6.900000, 3.100000, 5.400000, 2.100000, },
{6.700000, 3.100000, 5.600000, 2.400000, },
{6.900000, 3.100000, 5.100000, 2.300000, },
{5.800000, 2.700000, 5.100000, 1.900000, },
{6.800000, 3.200000, 5.900000, 2.300000, },
{6.700000, 3.300000, 5.700000, 2.500000, },
{6.700000, 3.000000, 5.200000, 2.300000, },
{6.300000, 2.500000, 5.000000, 1.900000, },
{6.500000, 3.000000, 5.200000, 2.000000, },
{6.200000, 3.400000, 5.400000, 2.300000, },
{5.900000, 3.000000, 5.100000, 1.800000, },
};
	int member_class[150] = {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, 2.000000, };
	
    int k = 5;
    Point distance[150];
  
	//kod do znalezienia k najbliższych
    for(int i = 0; i < 150; i++){
  	    double res = 0;
        for(int j = 0; j < 4; j++){
  		    res += pow2(x[j] - attributes[i][j]);
  	    }
        res = sqrt(res);
        Point res_dist;
        res_dist.idx = i;
        res_dist.distance = res;
        distance[i] = res_dist;
    }
  
    quick_sort(distance, 0, 150-1);
  
	//znalezienie najliczniejszej klasy
    int class_count[3] = {0.000000, 0.000000, 0.000000, }; //[0,0,0,...,0]
    for(int i = 0; i < k; i++){
        class_count[member_class[distance[i].idx]]++;
    }
  
    int max_count = -1;
    int idx_max = -1;
    for(int i = 0; i < 3; i++){
        if(class_count[i] > max_count){
            idx_max = i;
            max_count = class_count[i];
        }
    }
  
  return idx_max;
  
}
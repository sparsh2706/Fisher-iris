#include <bits/stdc++.h> 
using namespace std; 

int setbit(int s){
	int sum =0;
	int val =0;
	for (int i = 2; i >0; ++i)
	{
		sum += pow(2,i);
		if (sum > s)
		{
			val = i-1;
			sum = sum-pow(2,i);
			break;
		}
	}
	int va = pow(2,val);
	int value;
	while(true){
		value = __builtin_popcountll(va);
		sum += value;
		if (sum >= s)
		{
			return va;
		}
		va++;
	}
}

int main(){
	int ans = setbit(100);
	cout << ans;
}

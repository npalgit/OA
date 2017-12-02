package design;

/**
 * Created by tao on 10/18/17.
 */
public class Heap {

    public int []nums = null;

    public Heap(int []arr){
        int n = arr.length;
        nums =  new int[n+1];
        for(int i=0;i<n;++i)
            nums[i+1]=arr[i];
    }

    public int getRight(int ind){
        return ind*2+1;
    }

    public int getLeft(int ind){
        return ind*2;
    }

    public int getParent(int ind){
        return ind/2;
    }

    public void heapify(int ind,int upper){
        int l = getLeft(ind);
        int r = getRight(ind);
        int minInd = ind;
        if(l<upper && nums[minInd]>nums[l])
            minInd = l;
        if(r<upper && nums[minInd]>nums[r])
            minInd = r;
        if(minInd!=ind){
            int tmp = nums[minInd];
            nums[minInd]=nums[ind];
            nums[ind]=tmp;
            heapify(minInd,upper);
        }
    }

    public void buildHeap(int len){
        int n = len/2;
        for(int i=n;i>=1;--i)
            heapify(i,len+1);
    }

    public int peek(){
        if(nums.length<2)
            throw new IndexOutOfBoundsException();
        return nums[1];
    }

    public int[] sort(){
        int n = nums.length,ind=0;
        buildHeap(n-1);
        int [] ans = new int[n-1];
        while(ind<n-1){
            ans[ind++]=nums[1];
            nums[1]=nums[n-ind];
            buildHeap(n-ind-1);
        }
        return ans;
    }
}

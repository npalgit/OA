package hard;

import commons.Point;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;

/**
 * Created by tao on 10/6/17.
 */
public class HardQuestions {
    public int[] smallestRange(List<List<Integer>> nums) {
        int minx=0, miny= Integer.MAX_VALUE, max_i=Integer.MIN_VALUE, min_i=0;
        int n = nums.size();
        int []next= new int[n];
        PriorityQueue<Integer>pq = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer i, Integer j) {
                return nums.get(i).get(next[i])-nums.get(j).get(next[j]);
            }
        });
        for(int i=0;i<n;++i){
            max_i=Math.max(max_i,nums.get(i).get(0));
            pq.offer(i);
        }
        boolean flag = true;
        while(!pq.isEmpty() && flag){
            min_i = pq.poll();
            if(miny-minx>max_i-nums.get(min_i).get(next[min_i])){
                miny = max_i;
                minx = nums.get(min_i).get(next[min_i]);
            }
            if(next[min_i]==nums.get(min_i).size()-1){
                flag= false;
                break;
            }
            next[min_i]++;
            pq.offer(min_i);
            max_i = Math.max(max_i,nums.get(min_i).get(next[min_i]));
        }
        return new int[]{minx,miny};
    }

    public int highBound(int target,int row,int n){
        int ans = 0;
        int start = 0, end =n-1, mid=0;
        if(target>=(row+1)*n)
            return n;
        while(start<=end){
            mid = (end-start)/2+start;
            if((row+1)*(mid+1)<=target)
                start = mid+1;
            else
                end = mid-1;
        }
        return start;
    }

    public int lowBound(int target,int row,int n){
        int ans = 0;
        int start =0 , end =n-1,mid=0;
        if(target>(row+1)*n)
            return n;
        while(start<=end){
             mid = (end-start)/2+start;
            if((row+1)*(mid+1)>=target)
                end = mid-1;
            else
                start = mid+1;
        }
        return start;
    }
    public int[] getSmaller(int i,int j,int m,int n){
        int []res={j+1,j+1};
        //start from i-1 to search
        for(int row =i-1;row>=0;--row){
            res[0]+=lowBound((i+1)*(j+1),row,n);
            res[1]+=highBound((i+1)*(j+1),row,n);
        }
        //start from i+1 to search
        for(int row=i+1;row<m;++row){
            res[0]+=lowBound((i+1)*(j+1),row,n);
            res[1]+=highBound((i+1)*(j+1),row,n);
        }
        return res;

    }
    public Map<String,Integer>map = new HashMap<>();
    public Map<Integer,int[]>map1 = new HashMap<>();
    public int findKthNumber(int minRow,int maxRow,int minCol,int maxCol,int k,int m,int n){
        if(minRow>maxRow||minCol>maxCol||minRow<0||minCol<0||maxCol>=n||maxRow>=m)
            return -1;
        StringBuilder sb = new StringBuilder();
        sb.append(minRow+1).append(maxRow+1).append(minCol+1).append(maxCol+1);
        if(map.containsKey(sb.toString()))
            return map.get(sb.toString());
        int row = (maxRow-minRow)/2+minRow;
        int col = (maxCol-minCol)/2+minCol;
        int []num={0,0};
        if(map1.containsKey((row+1)*(col+1)))
            num = map1.get((row+1)*(col+1));
        else{
            num =getSmaller(row,col,m,n);
            map1.put((row+1)*(col+1),num);
        }
        if(k>=num[0] && k<=num[1]){
                map.put(sb.toString(),(row+1)*(col+1));
                return (row+1)*(col+1);
        }
        else if(k>num[1]){
            sb.setLength(0);
            sb.append(row+1).append(maxRow+1).append(minCol+1).append(maxCol+1);
            int val =0;
            if(map.containsKey(sb.toString()))
                val = map.get(sb.toString());
            else{
                val=findKthNumber(row+1,maxRow,minCol,maxCol,k,m,n);
                map.put(sb.toString(),val);
            }
            if(val!=-1){
                return val;
            }
            else{
                sb.setLength(0);
                sb.append(minRow+1).append(maxRow+1).append(col+1).append(maxCol+1);
                if(map.containsKey(sb.toString()))
                    val = map.get(sb.toString());
                else{
                    val = findKthNumber(minRow,maxRow,col+1,maxCol,k,m,n);;
                    map.put(sb.toString(),val);
                }
                return val;
            }
        }
        else{
            sb.setLength(0);
            sb.append(minRow+1).append(row-1).append(minCol+1).append(maxCol+1);
            int val =0;
            if(map.containsKey(sb.toString())){
                val = map.get(sb.toString());
            }else{
                val = findKthNumber(minRow,row-1,minCol,maxCol,k,m,n);
                map.put(sb.toString(),val);
            }
            if(val!=-1)
                return val;
            else{
                sb.setLength(0);
                sb.append(minRow+1).append(maxRow+1).append(minCol+1).append(col-1);
                if(map.containsKey(sb.toString()))
                    val = map.get(sb.toString());
                else{
                    val = findKthNumber(minRow,maxRow,minCol,col-1,k,m,n);
                    map.put(sb.toString(),val);
                    return val;
                }
            }
            return val;
        }
    }
    public int findKthNumber(int m, int n, int k) {



        return findKthNumber(0,m-1,0,n-1,k,m,n);



//        int start = 0, end = m*n-1,mid=0;
//        while(start<=end){
//             mid = (end-start)/2+start;
//            int i = mid/n;
//            int j = mid%n;
//            int []num =getSmaller(i,j,m,n);
//            if(k>=num[0] && k<=num[1])
//                return (i+1)*(j+1);
//            else if(num[1]<k)
//                start =  mid+1;
//            else
//                end = mid-1;
//        }
//        int val = (start/n+1)*(start%n+1);
//        int x = start/n,y = start%n;
//        int []num={0,0};
//        if(x>=1){
//            num = getSmaller(x-1,y,m,n);
//            if(k>=num[0] && k<=num[1])
//                return x*(y+1);
//        }
//        if(y>=1){
//            num = getSmaller(x,y-1,m,n);
//            if(k>=num[0] && k<=num[1])
//                return (x+1)*y;
//        }
//        while(x>=1 && y<n-1 && (x+1)*(y+1)>x*(y+2)){
//            num = getSmaller(x-1,y+1,m,n);
//            if(k>=num[0] && k<=num[1])
//                return x*(y+2);
//            else if(num[0]>k)
//                break;
//            y++;
//        }
//        while(y>=1 && x<m-1 && (x+1)*(y+1)>y*(x+2)){
//            num = getSmaller(x+1,y-1,m,n);
//            if(k>=num[0] && k<=num[1])
//                return y*(x+2);
//            else if(num[0]>k)
//                break;
//            x++;
//        }
//        return val-1;
    }

    public double getCos(Point o1,Point p){
        if(o1.x==p.x && o1.y==p.y)
            return 1.0;
        return (o1.x-p.x)*1.0/(Math.sqrt((o1.x-p.x)*(o1.x-p.x)+(o1.y-p.y)*(o1.y-p.y)));
    }

    public int getCross(Point a,Point b,Point c){
        int x1=b.x-a.x;
        int x2 = c.x-b.x;
        int y1 = b.y-a.y;
        int y2 = c.y-b.y;
        return x1*y2-x2*y1;
    }
    public List<Point> outerTrees(Point[] points) {
        //find the lowest bottom left
        Point p = new Point(Integer.MAX_VALUE, Integer.MAX_VALUE);
        for(Point point:points){
            if(p.y>point.y||p.y==point.y && p.x>point.x){
                p.x = point.x;
                p.y = point.y;
            }
        }

        //sort by angle;
        Arrays.sort(points, new Comparator<Point>() {
            @Override
            public int compare(Point o1, Point o2) {
                double cos1 = getCos(o1,p);
                double cos2 = getCos(o2,p);
                cos1 = BigDecimal.valueOf(cos1)
                        .setScale(4, RoundingMode.HALF_UP)
                        .doubleValue();
                cos2 = BigDecimal.valueOf(cos2)
                        .setScale(4, RoundingMode.HALF_UP)
                        .doubleValue();
                //System.out.println(cos1);
                if(cos1!=cos2)
                    return cos2>cos1?1:-1;
                else if(o2.y!=o1.y)
                    return o2.y-o1.y;
                else
                    return o1.x-o2.x;
            }
        });
        List<Point>res = new ArrayList<>();
        int negative =0;
        int positive =0;
        int n = points.length;
        for(int i=0;i<n;++i){
            if(res.size()<3)
                res.add(points[i]);
            else{
                //judge the last is valid or not
                int nn = res.size();
                int val1 = getCross(res.get(nn-3),res.get(nn-2),res.get(nn-1));
                if(val1>0)
                    positive++;
                if(val1<0)
                    negative++;
                int val2 = getCross(res.get(nn-2),res.get(nn-1),points[i]);
                if(val2>0)
                    positive++;
                if(val2<0)
                    negative++;

                if(val1*val2<0||negative*positive!=0){
                    res.remove(nn-1);
                    if(val1>0)
                        positive--;
                    if(val1<0)
                        negative--;
                    if(val2>0)
                        positive--;
                    if(val2<0)
                        negative--;
                    i--;
                }else
                    res.add(points[i]);
            }
        }
        System.out.println(res.size());
        return res;
    }


//    class SegmentTreeNode {
//        public int start, end, max;
//        public SegmentTreeNode left, right;
//        public SegmentTreeNode(int start, int end, int max) {
//            this.start = start;
//            this.end = end;
//            this.max = max;
//            this.left = this.right = null;
//        }
//    }
//
//    public SegmentTreeNode build(int[]A,int start,int end){
//        if(start>end)
//            return null;
//        if(start==end)
//            return new SegmentTreeNode(start,end,A[start]);
//        SegmentTreeNode root=new SegmentTreeNode(start,end,0);
//        int mid =(end-start)/2+start;
//        root.left=build(A,start,mid);
//        root.right=build(A,mid+1,end);
//        root.max=Math.max(root.left.max,root.right.max);
//        return root;
//    }
//
//    public int[] query(SegmentTreeNode root,int start,int end){
//        int []res={0,0};
//        if(root.end<start||root.start>end){
//            res[0]=  -0x7fffffff;
//            res[1]=-1;
//            return res;
//        }
//        if(root.start==root.end){
//            res[0]=root.max;
//            res[1]=root.start;
//            return res;
//        }
//        if(start>=root.right.start)
//            return query(root.right,start,end);
//        else if(end<=root.left.end)
//            return query(root.left,start,end);
//        else{
//            res=query(root.left,start,root.left.end);
//            int[]res1=query(root.right,root.right.start,end);
//            if(res[0]!=res1[0])
//                return res[0]>res1[0]?res:res1;
//            else
//                return res[1]<res1[1]?res:res1;
//        }
//    }


    int st[]; //array to store segment tree



    // A utility function to get the middle index from corner
    // indexes.
    int getMid(int s, int e) {
        return s + (e - s) / 2;
    }

    /*  A recursive function to get the minimum value in a given
        range of array indexes. The following are parameters for
        this function.

        st    --> Pointer to segment tree
        index --> Index of current node in the segment tree. Initially
                   0 is passed as root is always at index 0
        ss & se  --> Starting and ending indexes of the segment
                     represented by current node, i.e., st[index]
        qs & qe  --> Starting and ending indexes of query range */
    int []RMQUtil(int ss, int se, int qs, int qe, int index)
    {
        // If segment of this node is a part of given range, then
        // return the min of the segment
        int []res ={0,0};
        if (qs <= ss && qe >= se){
            res[0]=st[index];
            res[1]=index;
            return res;
        }

        // If segment of this node is outside the given range
        if (se < qs || ss > qe){
            res[0]=Integer.MIN_VALUE;
            res[1]=-1;
            return res;
        }

        // If a part of this segment overlaps with the given range
        int mid = getMid(ss, se);
        int []left = RMQUtil(ss, mid, qs, qe, 2 * index + 1);
        int []right =  RMQUtil(mid + 1, se, qs, qe, 2 * index + 2);
        return left[0]<right[0]?right:left;
    }

    // Return minimum of elements in range from index qs (quey
    // start) to qe (query end).  It mainly uses RMQUtil()
    int []RMQ(int n, int qs, int qe)
    {
        // Check for erroneous input values
        int []res ={0,0};
        if (qs < 0 || qe > n - 1 || qs > qe) {
            res[0]=Integer.MIN_VALUE;
            res[1]=-1;
            return res;
        }

        return RMQUtil(0, n - 1, qs, qe, 0);
    }

    // A recursive function that constructs Segment Tree for
    // array[ss..se]. si is index of current node in segment tree st
    int constructSTUtil(int arr[], int ss, int se, int si)
    {
        // If there is one element in array, store it in current
        //  node of segment tree and return
        if (ss == se) {
            st[si] = arr[ss];
            return arr[ss];
        }

        // If there are more than one elements, then recur for left and
        // right subtrees and store the minimum of two values in this node
        int mid = getMid(ss, se);
        st[si] = Math.max(constructSTUtil(arr, ss, mid, si * 2 + 1),
                constructSTUtil(arr, mid + 1, se, si * 2 + 2));
        return st[si];
    }

    /* Function to construct segment tree from given array. This function
       allocates memory for segment tree and calls constructSTUtil() to
       fill the allocated memory */
    void constructST(int arr[], int n)
    {
        // Allocate memory for segment tree

        //Height of segment tree
        int x = (int) (Math.ceil(Math.log(n) / Math.log(2)));

        //Maximum size of segment tree
        int max_size = 2 * (int) Math.pow(2, x) - 1;
        st = new int[max_size]; // allocate memory

        // Fill the allocated memory st
        constructSTUtil(arr, 0, n - 1, 0);
    }

    public void setup(int[][]dp,int[]arr,int i){
        dp[i][0]=i;
        dp[i][1]=arr[1];
        dp[i][2]+=arr[0];
    }
    public int[] maxSumOfThreeSubarrays(int[] nums, int k) {
        // List<Tuple>res = new ArrayList<>();
        int []ans ={0,0,0};
        int sum = 0,n=nums.length;
        int []A = new int[n-k+1];
        int ind =0;
        for(int i=0;i<n;++i){
            sum+=nums[i];
            if(i>=k-1){
                A[ind++]=sum;
                sum-=nums[i-k+1];
            }
        }
        constructST(A,A.length);
        int nn = A.length;
        int[][]dp = new int[nn][3];
        for(int i=0;i<nn;++i){
            int []left = RMQ(nn,0,i-k);
            int []right = RMQ(nn,i+k,nn-1);
            if(right[0]>left[0]||right[0]==left[0] && right[1]<left[1]){
                setup(dp,right,i);
            }else
                setup(dp,left,i);
        }

        int maxVal = 0;
        for(int i=0;i<nn;++i){
            int []point = {dp[i][0],dp[i][0]+k-1,dp[i][1],dp[i][1]+k-1};
            Arrays.sort(point);
            int []left = RMQ(nn,0,point[0]-k);
            int []middle = RMQ(nn,point[1]+k,point[2]-k);
            int []right = RMQ(nn,point[2]+k,nn-1);
            int val = dp[i][1];
            if(left[0]>Math.max(middle[0],right[0])||left[0]==Math.max(middle[0],right[0]) && left[1]<Math.min(right[1],middle[1])){
                setup(dp,left,i);
            }else if(right[0]>Math.max(middle[0],left[0])||right[0]==Math.max(middle[0],left[0]) && right[1]<Math.min(left[1],middle[1])){
                setup(dp,right,i);
            }else{
                setup(dp,middle,i);
            }
            if(maxVal<A[i]+dp[i][2]||maxVal==A[i]+dp[i][2] && Math.min(Math.min(val,dp[i][1]),i)<Math.min(Math.min(ans[0],ans[1]),ans[2])){
                ans[0]=i;
                ans[1]=val;
                ans[2]=dp[i][1];
                maxVal = A[i]+dp[i][2];
            }
        }
        Arrays.sort(ans);
        return ans;

    }


    public String convert(List<int[]>small,int x,int y){
        StringBuilder sb = new StringBuilder();
        int n = small.size();
        sb.append(n).append("@");
        for(int i=0;i<n;++i){
            small.set(i,new int[]{small.get(i)[0]-x,small.get(i)[1]-y});
        }
        Collections.sort(small,new Comparator<int[]>(){
            public int compare(int[]arr1,int []arr2){
                if(arr1[0]!=arr2[0])
                    return arr1[0]-arr2[0];
                else
                    return arr1[1]-arr2[1];
            }
        });
        for(int i=0;i<n;++i){
            sb.append(small.get(i)[0]).append("#").append(small.get(i)[1]).append("@");
        }
        return sb.toString();
    }
    public int numDistinctIslands(int[][] matrix) {
        if(matrix.length==0||matrix[0].length==0)
            return 0;
        int m = matrix.length,n = matrix[0].length;
        boolean [][]vis = new boolean[m][n];
        int []dx={1,-1,0,0};
        int []dy ={0,0,1,-1};
        Set<String>islands = new HashSet<>();
        Queue<int[]>q=new LinkedList<>();
        for(int i=0;i<m;++i){
            for(int j=0;j<n;++j){
                if(matrix[i][j]==1 && !vis[i][j]){
                    q.offer(new int[]{i,j});
                    vis[i][j]=true;
                    int size=0;
                    List<int[]>small = new ArrayList<>();
                    small.add(new int[]{i,j});
                    int topX=i, topY = j;
                    while(!q.isEmpty()){
                        int []top = q.poll();
                        size++;
                        int x = top[0];
                        int y =top[1];
                        for(int k=0;k<4;++k){
                            int nx = x+dx[k];
                            int ny = y+dy[k];
                            if(nx<0||nx>=m||ny<0||ny>=n||vis[nx][ny]||matrix[nx][ny]!=1)
                                continue;
                            vis[nx][ny]=true;
                            q.offer(new int[]{nx,ny});
                            small.add(new int[]{nx,ny});
                            if(topX>nx||topX==nx && topY>ny){
                                topX=nx;
                                topY=ny;
                            }
                        }
                    }
                    islands.add(convert(small,topX,topY));

                }

            }
        }
        return islands.size();
    }


    public void transform(StringBuilder board){
        int ii=0;
        boolean hasNext = false;
        while(!hasNext &&board.length()!=0){
            ii=1;
            int cnt=1;
            while(ii<=board.length()){
                if(ii<board.length() && board.charAt(ii)==board.charAt(ii-1))
                    cnt++;
                else{
                    if(cnt>=3){
                        board.delete(ii-cnt,ii);
                        hasNext=true;
                        break;
                    }else{
                        cnt=1;
                    }
                }
                ii++;
            }

            if(hasNext)
                hasNext=false;
            else
                hasNext=true;
        }
    }
    Map<String,Integer>map11 = new HashMap<>();
    Map<Character,Integer>map111=new HashMap<>();
    public int findMinStep(StringBuilder board,int []d){
        //transform
        transform(board);
        if(board.length()==0)
            return 0;
        String[]ss={"R","Y","B","G","W"};
        int cnt = 0;
        boolean has=false;
        for(int i=0;i<5;++i){
            cnt=10*cnt+d[i];
            if(d[i]>0 && board.indexOf(ss[i])>-1){
                has=true;
                //break;
            }
        }
        String xxx  = board.toString()+cnt;
        if(map11.containsKey(xxx))
            return map11.get(xxx);
        int val = 0x7fffffff;
        if(cnt==0||!has){
            map11.put(xxx,0x7fffffff);
            return 0x7fffffff;
        }
        int n = board.length();
        for(int i=0;i<n;++i){
            int []offset={i-1,i};
            for(int j=0;j<2;++j){
                if(offset[j]<0)
                    continue;
                int ind=0;
                try{
                    ind = map111.get(board.charAt(offset[j]));
                }catch (Exception e){
                    System.out.println(ind);
                    System.out.println(board.charAt(offset[j]));
                    System.exit(0);
                }
                if(d[ind]<=0)
                    continue;
                StringBuilder sb = new StringBuilder(board);
                sb.insert(i,ss[ind]);
                d[ind]--;
                int next = findMinStep(sb,d);
                d[ind]++;
                if(next==0x7fffffff)
                    continue;
                else
                    val=Math.min(val,1+next);
            }
        }
        map11.put(xxx,val);
        return val;
    }
    public int findMinStep(String board, String hand) {
        int[]d =new int[5];
        int n = hand.length();
        for(int i=0;i<n;++i){
            if(hand.charAt(i)=='R')
                d[0]++;
            else if(hand.charAt(i)=='Y')
                d[1]++;
            else if(hand.charAt(i)=='B')
                d[2]++;
            else if(hand.charAt(i)=='G')
                d[3]++;
            else
                d[4]++;
        }
        map111.put('R',0);
        map111.put('Y',1);
        map111.put('B',2);
        map111.put('G',3);
        map111.put('W',4);
        StringBuilder sb = new StringBuilder(board);
        int steps = findMinStep(sb,d);
        return steps==Integer.MAX_VALUE?-1:steps;
    }


    //O(N)解法居然不行，只能用O(lgn)
    //这个思路是非常棒的
    public int findIntegers(int num) {
        return find(0,0,num,true);
    }
    private int find(int i,int sum,int num,boolean validSuffix){
        if(sum>num)
            return 0;
        if(1<<i >num)
            return 1;
        if(!validSuffix)
            return find(i+1,sum,num,true);
        return find(i+1,sum,num,true)+find(i+1,sum+(1<<i),num,false);
    }

    public int findIntegersByDP(int num){
        StringBuilder sb = new StringBuilder(Integer.toBinaryString(num));
        int n = sb.length();
        int []a = new int[n];
        int []b = new int[n];
        a[0]=b[0]=1;
        for(int i=1;i<n;++i){
            a[i]=a[i-1]+b[i-1];
            b[i]=a[i-1];
        }
        int sum = a[n-1]+b[n-1];
        for(int i=1;i<n;++i){
            if(sb.charAt(i)=='1' && sb.charAt(i-1)=='1')
                break;
            if(sb.charAt(i)=='0' && sb.charAt(i-1)=='0')
                sum-=b[n-i-1];
        }
        return sum;
    }


}

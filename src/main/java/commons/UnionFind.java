package commons;

/**
 * Created by tao on 10/21/17.
 */
public class UnionFind {
    public int []parent=null;
    public int[]rank=null;
    public UnionFind(int n){
        parent=new int[n+1];
        rank=new int[n+1];
        for(int i=0;i<=n;++i)
            parent[i]=i;
    }
    public int find(int x){
        while(x!=parent[x]){
            parent[x]=parent[parent[x]];
            x=parent[x];
        }
        return x;
    }
    public boolean mix(int x,int y){
        int xx = find(x);
        int yy = find(y);
        if(xx==yy)
            return false;
        if(rank[xx]<rank[yy]){
            parent[xx]=yy;
        }else if(rank[yy]<rank[xx]){
            parent[yy]=xx;
        }else{
            parent[xx]=yy;
            rank[yy]++;
        }
        return true;
    }
}

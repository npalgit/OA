package commons;

/**
 * Created by tao on 10/21/17.
 */
public class ReadN extends Read4 {
    public int cnt=0;
    public int curEnd=0;
    public char []buffer=new char[4];
    public int read(char[] buf, int n) {
        int res=0;
        boolean hasNext=true;
        while(res<n && hasNext){
            //only if we there is no word in last time
            if(curEnd==0)
                cnt=read4(buffer);
            if(cnt<4)
                hasNext=false;
            for(;curEnd<cnt && res<n;++curEnd)
                buf[res++]=buffer[curEnd];
            if(curEnd==cnt)
                curEnd=0;
        }
        return Math.min(res,n);
    }
}

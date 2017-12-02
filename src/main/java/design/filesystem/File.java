package design.filesystem;

/**
 * Created by tao on 10/24/17.
 */
public class File extends Entry{

    private String content;
    private int size;

    public File(String name, Directory p, int sz){
        super(name,p);
        size =sz;
    }

    public int size(){
        return size;
    }

    public String getContents(){
        return content;
    }

    public void setContent(String c){
        content =c;
    }
}

package design.filesystem;

import java.util.ArrayList;

/**
 * Created by tao on 10/24/17.
 */
public class Directory extends Entry {

    protected ArrayList<Entry> contents;
    protected Directory(String name, Directory p){
        super(name,p);
        contents = new ArrayList<>();
    }

    public ArrayList<Entry>getContents(){
        return contents;
    }

    public int size(){
        int size = 0;
        for(Entry e:contents){
            size +=e.size();
        }
        return size;
    }

    public int numberOfFiles(){
        int count = 0;
        for(Entry e:contents){
            if(e instanceof Directory){
                count++;
                Directory d = (Directory)e;
                count +=d.numberOfFiles();
            }else if(e instanceof File){
                count++;
            }
        }
        return count;
    }
    public boolean deleteEntry(Entry entry){
        return contents.remove(entry);
    }

    public void addEntry(Entry entry){
        contents.add(entry);
    }
}

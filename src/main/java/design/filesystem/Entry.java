package design.filesystem;

/**
 * Created by tao on 10/24/17.
 */
public abstract class Entry {
    protected Directory parent;
    protected long created;
    protected long lastUpdated;
    protected long lastAccessed;
    protected String name;

    public Entry(String name, Directory p){
        this.name = name;
        this.parent = p;
        created = System.currentTimeMillis();
    }

    public boolean delete(){
        if(parent ==null){
            return false;
        }

        return parent.deleteEntry(this);
    }

    public abstract  int size();

    public String getFullPath(){
        if(parent ==null)
            return name;
        else
            return parent.getFullPath()+"/"+name;
    }

    public long getCreationTime(){
        return created;
    }

    public long getLastUpdatedTime(){
        return lastUpdated;
    }

    public long getLastAccessed(){
        return lastAccessed;
    }

    public void changeName(String name){
        this.name = name;
    }
    public String getName(){
        return name;
    }


}

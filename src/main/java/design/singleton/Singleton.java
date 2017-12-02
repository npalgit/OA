package design.singleton;

/**
 * Created by tao on 10/14/17.
 */
public class Singleton {

    private static Singleton singleton;
    private Singleton (){

    }

    public Singleton getSingleton(){
        if(singleton==null)
            singleton = new Singleton();
        return singleton;
    }

    //safe way
    public Singleton getSingletonSafe(){
        if(singleton==null){
            synchronized (Singleton.class){
                if(singleton==null)
                    singleton=new Singleton();
            }
        }
        return singleton;
    }
}

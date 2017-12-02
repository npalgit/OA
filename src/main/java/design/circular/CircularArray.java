package design.circular;

import java.util.Iterator;

/**
 * Created by tao on 10/24/17.
 */
public class CircularArray<T> implements Iterable<T> {
    private T [] items;
    private int head = 0;

    public CircularArray(int size){
        items = (T[]) new Object[size];
    }

    public int convert(int index){
        if(index<0)
            index+=items.length;
        return (head+index)%items.length;
    }

    public void rotate(int shiftRight){
        head = convert(shiftRight);
    }

    public T get(int i){
        if(i<0||i>=items.length)
            throw new IndexOutOfBoundsException("Index "+i+ " is out of bounds");
        return items[convert(i)];
    }

    public void set(int i, T item){
        items[convert(i)]=item;
    }

    public Iterator<T> iterator(){
        return new CircularArrayIterator();
    }

    private class CircularArrayIterator implements Iterator<T>{
        private int _current =-1;
        public CircularArrayIterator(){};

        public boolean hasNext(){
            return _current< items.length-1;
        }

        public T next(){
            _current++;
            return (T)items[convert(_current)];
        }

        public void remove(){
            throw new UnsupportedOperationException("Remove is not supported by circularArray");
        }
    }

}

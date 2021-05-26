package org.jvmpy.python;

public class IndexedImpl<E> implements Indexed<E> {

	private E element;
	private int index;
	
	public static <T> Indexed<T> create(T element, int index) {
		return new IndexedImpl<>(element, index);
	}
	
	public IndexedImpl(E element, int index) {
		this.element = element;
		this.index = index;
	}
	
	@Override
	public E get() {
		return element;
	}

	@Override
	public int getIndex() {
		return index;
	}

}

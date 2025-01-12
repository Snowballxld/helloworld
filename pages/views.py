# pages/views.py
import pdb

from django.shortcuts import render, HttpResponseRedirect
from django.http import Http404
from django.urls import reverse
from django.views.generic import TemplateView

def homePageView(request):
    return render(request, 'home.html', {

        'mynumbers':[1,2,3,4,5,6,],

        'firstName': 'John',

        'lastName': 'Guo'
    })

def aboutPageView(request):
    # return request object and specify page.
    return render(request, 'about.html')

def JohnPageView(request):
    return render(request, 'John.html')


def homePost(request):
    # Use request object to extract choice.

    choice = -999

    gmat = -999  # Initialize gmat variable.

    try:

        # Extract value from request object by control name.

        currentChoice = request.POST['choice']

        gmatStr = request.POST['gmat']

        print("Just before Johns's breakpoint")

        #pdb.set_trace()

        #breakpoint()

        print("Just after breakpoint")

        # Crude debugging effort.

        print("*** Years work experience: " + str(currentChoice))

        choice = int(currentChoice)

        gmat = float(gmatStr)

        # Enters 'except' block if integer cannot be created.

    except:

        return render(request, 'home.html', {

            'errorMessage': '*** The data submitted is invalid. Please try again.',

            'mynumbers': [1, 2, 3, 4, 5, 6, ]})

    else:

        # Always return an HttpResponseRedirect after successfully dealing

        # with POST data. This prevents data from being posted twice if a

        # user hits the Back button.

        return HttpResponseRedirect(reverse('results', kwargs={'choice': choice, 'gmat': gmat}, ))


def results(request, choice, gmat):
    print("*** Inside reults()")

    return render(request, 'results.html', {'choice': choice, 'gmat': gmat})